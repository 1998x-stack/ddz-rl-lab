"""Optional pairwise learner consumes only reviewer-approved v2 decisions."""
import random
import json
import pytest
torch=pytest.importorskip('torch')
from ddz_core.engine import GameEngine
from ddz_core.state import Phase
from ddz_env.baselines import reasonable_action_index
from ddz_quality.feedback import submit_feedback, approve_feedback
from ddz_training.trainer import DMCTrainer, TrainConfig

def test_reviewed_feedback_is_separate_from_unreviewed_suggestions(tmp_path):
    game=GameEngine(67001)
    rng=random.Random(11)
    for _ in range(500):
        if game.state.phase==Phase.FINISHED:
            break
        obs=game.observe(game.state.current_seat,event_limit=16)
        game.step(obs.legal_actions[reasonable_action_index(obs,rng)])
    assert game.state.phase==Phase.FINISHED
    gid='c'*24
    (tmp_path/(gid+'.json')).write_text(json.dumps(game.export_replay()),encoding='utf-8')
    alt=0 if game.action_history[0].kind.value!='CALL' else 1
    feedback=submit_feedback(tmp_path,gid,0,alt,'auction')
    conf=TrainConfig(episodes=1,eval_games=1,eval_every=1,hidden=16,
                     model_schema='ddz-candidate-mlp-v2',
                     reviewed_feedback_dir=str(tmp_path),output=str(tmp_path/'training'))
    trainer=DMCTrainer(conf)
    trainer._learn_reviewed_preferences()
    assert trainer.updates==0
    approve_feedback(tmp_path,feedback['id'],'operator')
    trainer._learn_reviewed_preferences()
    assert trainer.updates==1

def test_reviewed_v2_feedback_rejects_v1_network(tmp_path):
    with pytest.raises(ValueError, match='v2 encoder'):
        TrainConfig(reviewed_feedback_dir=str(tmp_path),model_schema='ddz-candidate-mlp-v1')
