"""The urgent exception must occur before unconditional teammate PASS."""
from dataclasses import replace
import random
import pytest
torch=pytest.importorskip('torch')
from ddz_core import counts_from_ranks, generate_plays, recognize_patterns, Action, ActionKind, GameEngine
from ddz_core.state import Phase
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle

def test_teammate_lead_pass_is_conditional_on_landlord_cards():
    env=GameEngine(12)
    old=env.observe(env.state.current_seat)
    lead=recognize_patterns(counts_from_ranks('K','K'))[0]
    hand=counts_from_ranks('A','A','3','3')
    legal=generate_plays(hand,lead)+(Action.simple(ActionKind.PASS),)
    common=dict(seat=1,current_seat=1,phase=Phase.PLAY,landlord=0,
                own_counts=hand,last_play=lead,last_play_counts=counts_from_ranks('K','K'),
                last_play_seat=2,legal_actions=legal,remaining_counts=(2,4,5))
    urgent=replace(old,**common)
    calm=replace(urgent,remaining_counts=(3,4,5))
    bundle=RoleBundle(NetworkSpec.v2(16))
    a=bundle.act(urgent,random.Random(7),0.0).local_index
    b=bundle.act(calm,random.Random(7),0.0).local_index
    assert legal[a].kind==ActionKind.PLAY
    assert legal[b].kind==ActionKind.PASS
