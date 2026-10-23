"""Local training improvements use only verifiable candidate preferences."""
import random
import pytest
torch=pytest.importorskip('torch')
from ddz_training.trainer import DMCTrainer, TrainConfig, Experience
from ddz_training.actors import label_samples
from ddz_training.replay_buffer import ReplayBuffer
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle

def test_deal_aware_rewards_are_normalized():
    items=[('bid', (0.0,), (0.0,), 0, 0), ('landlord',(0.0,), (0.0,),0,1)]
    result=label_samples(items, final_deal=1, truncated=False,
                         reward=(2.0,-1.0,-1.0),redeal_penalty=.2)
    assert [x.return_value for x in result]==[-.2,1.0]

def test_replay_backward_compatibility_and_preference_fields():
    r=ReplayBuffer(5)
    r.load_state_dict({'capacity':5,'seen':1,'entries':[('landlord',(1.0,),(2.0,),0,1.0)]})
    assert r.sample(1,random.Random(1))[0].preferred_action is None
    r.add([Experience('bid',(1.0,),(2.0,),0,1.0,(3.0,),(4.0,),'7:0')])
    q=ReplayBuffer(5)
    q.load_state_dict(r.state_dict())
    assert q.sample(2,random.Random(2))[-1].return_value in (1.0,)

def test_pairwise_margin_changes_parameters(tmp_path):
    config=TrainConfig(episodes=1,eval_games=1,eval_every=1,hidden=16,
                       output=str(tmp_path/'train'))
    t=DMCTrainer(config)
    s=tuple([0.0]*t.bundle.spec.state_dim)
    a=tuple([0.0]*t.bundle.spec.action_dim)
    b=tuple([1.0]*t.bundle.spec.action_dim)
    t.learn([Experience('landlord',s,a,0,-1.0, b,a,'7:0')])
    assert t.updates==1

def test_optional_paired_bid_teacher_trains_on_observed_inputs(tmp_path):
    config=TrainConfig(episodes=1,eval_games=1,eval_every=1,hidden=16,
                       bid_counterfactual_deals=1,output=str(tmp_path/'paired'))
    trainer=DMCTrainer(config)
    trainer._learn_bid_counterfactual([99041])
    assert trainer.updates==1
    with pytest.raises(ValueError,match='0..8'):
        TrainConfig(bid_counterfactual_deals=9)
