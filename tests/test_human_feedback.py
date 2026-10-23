"""Human review may propose alternatives only from a verified historical legal set."""
import random
import pytest
from ddz_core.engine import GameEngine
from ddz_env.baselines import reasonable_action_index
from ddz_core.state import Phase
from ddz_quality.feedback import submit_feedback, read_feedback, FeedbackError

def test_feedback_replay_binding_and_file_permissions(tmp_path):
    engine=GameEngine(6703)
    rng=random.Random(9)
    for _ in range(250):
        if engine.state.phase==Phase.FINISHED:
            break
        obs=engine.observe(engine.state.current_seat,event_limit=16)
        engine.step(obs.legal_actions[reasonable_action_index(obs,rng)])
    assert engine.state.phase==Phase.FINISHED
    gid='a'*24
    path=tmp_path/(gid+'.json')
    import json
    path.write_text(json.dumps(engine.export_replay()),encoding='utf-8')
    # The first bid has exactly two legal actions, so index 1 is an alternative
    chosen=engine.action_history[0]
    target=0 if chosen.kind.value!='CALL' else 1
    feedback=submit_feedback(tmp_path,gid,0,target,'auction')
    row=read_feedback(tmp_path,feedback['id'])
    assert row['review_status']=='unreviewed'
    assert row['suggested_index']==target
    assert row['expert_label'] is None and row['behavior_probability'] is None
    assert (tmp_path/'feedback'/(feedback['id']+'.json')).stat().st_mode & 0o777 == 0o600
    with pytest.raises(FeedbackError):
        submit_feedback(tmp_path,gid,0,9999,'auction')
    with pytest.raises(FeedbackError):
        submit_feedback(tmp_path,'../etc/passwd',0,0,'auction')
