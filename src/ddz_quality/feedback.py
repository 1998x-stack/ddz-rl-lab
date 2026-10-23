"""Private, append-only human feedback tied to an authoritative replay decision.

Submitted feedback is NOT training ground truth. An operator must independently
adjudicate competing alternatives before supplying reviewed preference pairs.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from ddz_core.actions import Action
from ddz_core.engine import GameEngine, replay
from ddz_core.roles import role_for_observation
from ddz_env.encoding import encode_observation_v2, encode_action_v2
from ddz_web.persistence import ArenaJournal

class FeedbackError(ValueError):
    pass

_ID = re.compile(r'^[0-9a-f]{24}$')
_REASONS = frozenset(('split','teamwork','endgame','auction','legality','other'))
_MAX_BYTES = 4_000_000

def _read_trace(directory: Path, game_id: str) -> tuple[dict, object]:
    if not _ID.fullmatch(game_id):
        raise FeedbackError('invalid game id')
    if directory.is_symlink() or not directory.is_dir():
        raise FeedbackError('unsafe replay directory')
    path = directory / (game_id + '.json')
    if path.exists():
        if path.is_symlink() or path.stat().st_size > _MAX_BYTES:
            raise FeedbackError('unsafe replay file')
        record=json.loads(path.read_text('utf-8'))
    else:
        journal=ArenaJournal(directory)
        session=journal.read_one(game_id)
        if session is None:
            raise FeedbackError('unavailable game session')
        env=ArenaJournal.restore_env(session)
        record=env.export_replay()
        record['model_ids']=session['model_ids']
        record['model_sha256']=session['model_sha256']
    verified=replay(record)
    return record,verified

def submit_feedback(replay_dir: str | Path, game_id: str, step_index: int,
                    alternative_index: int | None, reason: str) -> dict:
    if (type(step_index) is not int or step_index < 0 or step_index > 1200 or
            (alternative_index is not None and
             (type(alternative_index) is not int or alternative_index < 0)) or
            reason not in _REASONS):
        raise FeedbackError('invalid feedback request')
    directory=Path(replay_dir)
    record,verified=_read_trace(directory,game_id)
    actions=record.get('actions')
    if not isinstance(actions,list) or step_index >= len(actions):
        raise FeedbackError('step does not reference a committed action')
    engine=GameEngine(record['seed'],verified.rules)
    for encoded in actions[:step_index]:
        engine.step(Action.from_dict(encoded))
    seat=engine.state.current_seat
    if seat is None:
        raise FeedbackError('terminal state has no action')
    obs=engine.observe(seat,event_limit=16)
    chosen=Action.from_dict(actions[step_index])
    if chosen not in obs.legal_actions:
        raise FeedbackError('recorded chosen action is not legal')
    alternative=None
    if alternative_index is not None:
        if alternative_index >= len(obs.legal_actions):
            raise FeedbackError('alternative is outside legal candidate set')
        alternative=obs.legal_actions[alternative_index]
        if alternative==chosen:
            raise FeedbackError('preferred move equals original move')
    # A user report alone is not an approved preference; NULL remains unknown.
    selected=obs.legal_actions.index(chosen)
    rule_hash=hashlib.sha256(json.dumps(asdict(verified.rules),sort_keys=True,
                                       separators=(',',':')).encode()).hexdigest()
    row={'schema':'ddz-human-feedback-v1', 'id':secrets.token_hex(16),
         'game_id':game_id,'step_index':step_index,'deal_index':obs.deal_index,
         'seat':seat,'role':role_for_observation(obs),
         'reason':reason,'review_status':'unreviewed','rule_sha256':rule_hash,
         'model_id':record.get('model_ids',['unknown']*3)[seat],
         'chosen_index':selected,'chosen_action':chosen.to_dict(),
         'suggested_index':alternative_index,
         'suggested_action':alternative.to_dict() if alternative is not None else None,
         'behavior_probability':None,'expert_label':None,
         'observation_v2':list(encode_observation_v2(obs)),
         'chosen_vector_v2':list(encode_action_v2(obs,chosen)),
         'suggested_vector_v2':(list(encode_action_v2(obs,alternative))
                                if alternative is not None else None)}
    # Each report is immutable and private: no path for user-controlled filename,
    # no overwrite, no unreviewed data accidentally added to model replay.
    folder=directory/'feedback'
    if folder.is_symlink():
        raise FeedbackError('unsafe feedback directory')
    folder.mkdir(mode=0o700,exist_ok=True)
    os.chmod(folder,0o700)
    path=folder/(row['id']+'.json')
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(row,stream,sort_keys=True,ensure_ascii=False,allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {'id':row['id'],'status':'unreviewed','game_id':game_id,'step_index':step_index}

def read_feedback(replay_dir: str | Path, feedback_id: str) -> dict:
    if not re.fullmatch(r'[0-9a-f]{32}',feedback_id):
        raise FeedbackError('invalid feedback id')
    path=Path(replay_dir)/'feedback'/(feedback_id+'.json')
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 20_000:
        raise FeedbackError('feedback unavailable or unsafe')
    row=json.loads(path.read_text('utf-8'))
    if row.get('id')!=feedback_id or row.get('schema')!='ddz-human-feedback-v1':
        raise FeedbackError('corrupted feedback')
    return row

def approve_feedback(replay_dir: str | Path, feedback_id: str,
                     reviewer: str) -> dict:
    """Second, explicit local-operator adjudication. Never mutate raw reports."""
    if not isinstance(reviewer,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{2,40}',reviewer):
        raise FeedbackError('invalid reviewer identifier')
    folder=Path(replay_dir)
    raw=read_feedback(folder,feedback_id)
    if raw.get('suggested_action') is None:
        raise FeedbackError('cannot approve an issue without a preferred legal alternative')
    # Reconstruct source again; do not trust submitted vectors without replay check.
    record,verified=_read_trace(folder,raw['game_id'])
    engine=GameEngine(record['seed'],verified.rules)
    if raw['step_index']>=len(record['actions']):
        raise FeedbackError('source action not present')
    for a in record['actions'][:raw['step_index']]:
        engine.step(Action.from_dict(a))
    obs=engine.observe(engine.state.current_seat,event_limit=16)
    chosen=Action.from_dict(record['actions'][raw['step_index']])
    suggested=Action.from_dict(raw['suggested_action'])
    if (chosen not in obs.legal_actions or suggested not in obs.legal_actions or
            chosen==suggested or raw['chosen_action']!=chosen.to_dict()):
        raise FeedbackError('reviewed preference does not match verified source')
    review={'schema':'ddz-human-review-v1','feedback_id':feedback_id,
            'reviewer':reviewer,'status':'approved',
            'feedback_sha256':hashlib.sha256(
                (folder/'feedback'/(feedback_id+'.json')).read_bytes()).hexdigest()}
    review_path=folder/'feedback'/(feedback_id+'.review.json')
    fd=os.open(review_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as stream:
        json.dump(review,stream,sort_keys=True)
        stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    return {'id':feedback_id,'status':'approved','reviewer':reviewer}

def approved_feedback_pairs(replay_dir: str | Path, *, limit: int=512) -> list[dict]:
    """Return only reviewed, hash-matched paired vectors; never infer expertise."""
    folder=Path(replay_dir)/'feedback'
    if folder.is_symlink() or not folder.is_dir() or not 1<=limit<=10000:
        return []
    rows=[]
    for path in sorted(folder.glob('*.review.json'))[:limit]:
        if path.is_symlink() or path.stat().st_size>3000:
            raise FeedbackError('unsafe review file')
        review=json.loads(path.read_text('utf-8'))
        if review.get('schema')!='ddz-human-review-v1' or review.get('status')!='approved':
            raise FeedbackError('invalid review status')
        fid=review['feedback_id']
        raw_path=folder/(fid+'.json')
        raw=read_feedback(Path(replay_dir),fid)
        if hashlib.sha256(raw_path.read_bytes()).hexdigest()!=review['feedback_sha256']:
            raise FeedbackError('feedback changed after operator review')
        if raw.get('suggested_vector_v2') is None:
            raise FeedbackError('missing reviewed preference vector')
        rows.append(raw)
    return rows
