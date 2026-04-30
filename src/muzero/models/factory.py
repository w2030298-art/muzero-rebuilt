"""Network factory that creates the appropriate model from configuration."""

from __future__ import annotations

import torch

from muzero.config.schema import AlgorithmConfig, NetworkConfig
from muzero.core.specs import EnvironmentSpec
from muzero.models.base import BaseMuZeroNetwork


class NetworkFactory:
    """Creates BaseMuZeroNetwork instances from configuration.

    Supports mlp, residual, and conv network types. The ``compile_model``
    option is off by default and only applied when explicitly enabled.
    """

    def create(
        self,
        config: NetworkConfig,
        env_spec: EnvironmentSpec,
        algo_config: AlgorithmConfig | None = None,
        compile_model: bool = False,
    ) -> BaseMuZeroNetwork:
        """Create a network instance matching the given configuration.

        Args:
            config: Network configuration specifying type, size, etc.
            env_spec: Environment specification (observation/action shape, player count).
            algo_config: Algorithm configuration (for EfficientZero flags).

        Returns:
            A ``BaseMuZeroNetwork`` instance.

        Raises:
            ValueError: If the network type is unknown.
        """
        model: BaseMuZeroNetwork

        if config.type == "mlp":
            # Determine action space properties
            action_size = env_spec.action.n if env_spec.action.n is not None else 1
            action_type = env_spec.action.type

            use_vp = algo_config.use_value_prefix if algo_config else False
            use_cl = algo_config.use_consistency_loss if algo_config else False

            from muzero.models.mlp import MLPNetwork

            model = MLPNetwork(
                observation_shape=env_spec.observation.shape,
                action_space_size=action_size,
                hidden_size=config.hidden_size,
                num_players=env_spec.num_players,
                use_value_prefix=use_vp,
                use_consistency_loss=use_cl,
                action_space_type=action_type,
                action_shape=env_spec.action.shape,
            )

        elif config.type == "residual":
            from muzero.models.residual import ResidualNetwork

            model = ResidualNetwork()

        elif config.type == "conv":
            from muzero.models.conv import ConvNetwork

            model = ConvNetwork()

        else:
            raise ValueError(f"Unknown network type: {config.type}. Supported: mlp, residual, conv")

        if compile_model:
            model = torch.compile(model)  # type: ignore[assignment]

        return model
