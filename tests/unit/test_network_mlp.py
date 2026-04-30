"""Tests for the MLP-based MuZero network."""

from __future__ import annotations

import torch

from muzero.models.mlp import MLPNetwork
from muzero.models.outputs import NetworkOutput


def test_initial_inference_shapes_single_player() -> None:
    """Verify initial inference produces correct tensor shapes."""
    model = MLPNetwork(
        observation_shape=(4,),
        action_space_size=2,
        hidden_size=128,
        num_players=1,
    )

    obs = torch.randn(8, 4)  # batch=8, obs_dim=4
    output = model.initial_inference(obs)

    assert isinstance(output, NetworkOutput)
    assert output.value.shape == (8,)
    assert output.reward.shape == (8,)
    assert output.policy_logits.shape == (8, 2)
    assert output.hidden_state.shape == (8, 128)
    assert output.value_prefix is None
    assert output.projection is None


def test_recurrent_inference_shapes_single_player() -> None:
    """Verify recurrent inference produces correct tensor shapes."""
    model = MLPNetwork(
        observation_shape=(4,),
        action_space_size=2,
        hidden_size=128,
        num_players=1,
    )

    hidden = torch.randn(8, 128)
    action = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    output = model.recurrent_inference(hidden, action)

    assert isinstance(output, NetworkOutput)
    assert output.value.shape == (8,)
    assert output.reward.shape == (8,)
    assert output.policy_logits.shape == (8, 2)
    assert output.hidden_state.shape == (8, 128)


def test_initial_inference_shapes_two_player() -> None:
    """Verify two-player network produces vector-valued outputs."""
    model = MLPNetwork(
        observation_shape=(9,),
        action_space_size=9,
        hidden_size=64,
        num_players=2,
    )

    obs = torch.randn(4, 9)
    output = model.initial_inference(obs)

    assert output.value.shape == (4, 2)
    assert output.reward.shape == (4, 2)
    assert output.policy_logits.shape == (4, 9)
    assert output.hidden_state.shape == (4, 64)


def test_get_set_weights_roundtrip() -> None:
    """Verify weights can be saved and restored correctly."""
    model = MLPNetwork(
        observation_shape=(4,),
        action_space_size=2,
        hidden_size=32,
        num_players=1,
    )

    # Get a reference output
    obs = torch.randn(3, 4)
    with torch.no_grad():
        ref_output = model.initial_inference(obs)

    # Save weights
    weights = model.get_weights()

    # Create a new model and load weights
    model2 = MLPNetwork(
        observation_shape=(4,),
        action_space_size=2,
        hidden_size=32,
        num_players=1,
    )
    model2.set_weights(weights)

    # Verify identical output
    with torch.no_grad():
        new_output = model2.initial_inference(obs)

    assert torch.allclose(ref_output.value, new_output.value)
    assert torch.allclose(ref_output.policy_logits, new_output.policy_logits)


def test_mlp_efficientzero_heads() -> None:
    """Verify EfficientZero heads produce correct shapes."""
    model = MLPNetwork(
        observation_shape=(4,),
        action_space_size=2,
        hidden_size=32,
        num_players=1,
        use_value_prefix=True,
        use_consistency_loss=True,
    )

    obs = torch.randn(4, 4)
    output = model.initial_inference(obs)

    assert output.value_prefix is not None
    assert output.value_prefix.shape == (4, 1)
    assert output.projection is not None
    assert output.projection.shape == (4, 128)


def test_mlp_continuous_action_dynamics() -> None:
    """Verify dynamics handles continuous action input."""
    model = MLPNetwork(
        observation_shape=(3,),
        action_space_size=1,
        hidden_size=32,
        num_players=1,
        action_space_type="continuous",
        action_shape=(1,),
    )

    hidden = torch.randn(4, 32)
    action = torch.randn(4, 1)  # Continuous action

    next_hidden, reward = model.dynamics(hidden, action)

    assert next_hidden.shape == (4, 32)
    assert reward.shape == (4,)


def test_initial_reward_is_zero() -> None:
    """Verify initial inference produces zero reward tensor."""
    model = MLPNetwork(
        observation_shape=(4,),
        action_space_size=2,
        hidden_size=32,
        num_players=1,
    )

    obs = torch.randn(2, 4)
    output = model.initial_inference(obs)

    assert torch.all(output.reward == 0.0)
