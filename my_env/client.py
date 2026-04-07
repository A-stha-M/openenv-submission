# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the Cold-Chain Dispatch OpenEnv environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MyEnvAction, MyEnvObservation


class MyEnv(EnvClient[MyEnvAction, MyEnvObservation, State]):
    """WebSocket-enabled client for running multi-step cold-chain episodes."""

    def _step_payload(self, action: MyEnvAction) -> Dict:
        """Serialize typed action for server step endpoint."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[MyEnvObservation]:
        """Parse server response payload into typed observation/result."""
        obs_data = payload.get("observation", {})
        if "done" not in obs_data:
            obs_data["done"] = payload.get("done", False)
        if "reward" not in obs_data:
            obs_data["reward"] = payload.get("reward", 0.0)

        observation = MyEnvObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse state endpoint response into OpenEnv State."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
