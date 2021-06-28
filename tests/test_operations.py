import hypothesis
import torch

from unitair import Field
from unitair.simulation.operations import apply_phase
from math import pi

from hypothesis import given
from tests.hypothesis_strategies.specialized import state_and_phase_angles


@given(state_and_phase_angles())
def test_apply_phase(data):
    angles = data['angles']
    original_state = data['state_vector']

    state = apply_phase(angles, original_state)
    state = apply_phase(-angles, state)
    assert state.isclose(original_state, atol=1e-4).all()

    complement_angles = 2 * pi - angles
    state = apply_phase(angles, original_state)
    state = apply_phase(complement_angles, state)
    # The weird atol here stops an issue with large angles ruining accuracy.
    # I think this is occurring because pi does not have sufficient precision.
    assert state.isclose(
        original_state,
        atol=1e-4 + angles.abs().sum() * .001
    ).all()


