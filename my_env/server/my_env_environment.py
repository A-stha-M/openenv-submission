# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Supply Chain Cold-Storage Rerouting Environment Implementation.

A real-world logistics environment where an agent must balance speed, 
cooling power, and fuel against dynamic ambient temperatures and traffic.
"""

import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    import models
    from models import MyEnvAction, MyEnvObservation
except ImportError:
    from ..models import MyEnvAction, MyEnvObservation


class MyEnvironment(Environment):
    """
    Cold-Chain Logistics Simulator.
    
    The agent controls a refrigerated truck. The cargo temperature rises
    based on ambient weather and drops based on cooling power. Higher speeds
    and higher cooling power drain fuel faster.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the logistics environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        
        # Internal physical state
        self.task_id = "cold_chain_easy"
        self.initial_dist = 0.0
        self.dist_dest = 0.0
        self.dist_hub = 0.0
        self.current_temp = 2.0
        self.ambient_temp = 25.0
        self.fuel = 100.0
        self.traffic_multiplier = 1.0
        self.cooling_health = 1.0
        self.current_location = "Origin"

    def reset(self, **kwargs) -> MyEnvObservation:
        """
        Reset the environment based on the task.
        Accepts dynamic task injection via kwargs from the HTTP API, 
        or falls back to environment variables.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        # Read the task difficulty injected by the automated evaluator baseline script
        self.task_id = kwargs.get("task_name", os.getenv("MY_ENV_TASK", "cold_chain_easy"))
        
        # Default starting stats
        self.current_temp = 2.0
        self.fuel = 100.0
        self.current_location = "En_Route"
        self.dist_hub = 150.0 

        # Configure task difficulty
        if self.task_id == "cold_chain_hard":
            self.dist_dest = 600.0
            self.ambient_temp = 40.0
            self.traffic_multiplier = 2.0 # Severe delays
            self.cooling_health = 0.8
        elif self.task_id == "cold_chain_medium":
            self.dist_dest = 400.0
            self.ambient_temp = 35.0
            self.traffic_multiplier = 1.2
            self.cooling_health = 0.6 # Partial cooling failure
        else: # cold_chain_easy
            self.dist_dest = 300.0
            self.ambient_temp = 25.0
            self.traffic_multiplier = 1.0
            self.cooling_health = 1.0
            
        self.initial_dist = self.dist_dest

        return self._generate_observation(reward=0.0, done=False)

    def step(self, action: MyEnvAction) -> MyEnvObservation:  # type: ignore[override]
        """
        Execute a 1-hour time step in the logistics simulation.
        """
        self._state.step_count += 1

        target = action.target_hub
        speed = action.speed_kmh
        cooling = action.cooling_power

        # 1. Calculate distance covered in this 1-hour step
        actual_speed = speed / self.traffic_multiplier
        distance_covered = actual_speed

        step_reward = 0.0
        done = False

        # 2. Update routing and calculate dense progress reward
        if target == "Destination":
            progress = min(distance_covered, self.dist_dest)
            self.dist_dest -= progress
            # Dense reward: Up to 0.5 total for covering the whole distance
            step_reward += (progress / self.initial_dist) * 0.5
        else: # Targeting Repair_Hub
            progress = min(distance_covered, self.dist_hub)
            self.dist_hub -= progress
            # Smaller progress reward for diverting
            step_reward += (progress / self.initial_dist) * 0.2

        # 3. Calculate Fuel Consumption
        # Base engine burn (exponential curve) + cooling unit burn
        fuel_burn = ((speed / 50.0) ** 2) * 2.0 + (cooling * 5.0)
        self.fuel -= fuel_burn

        # 4. Calculate Heat Decay (Simplified Newton's Law of Cooling)
        insulation_factor = 0.1
        cooling_effect = cooling * self.cooling_health * 5.0
        heat_gained = insulation_factor * (self.ambient_temp - self.current_temp)
        
        # Apply net temperature change
        self.current_temp += (heat_gained - cooling_effect)

        # 5. Evaluate Terminal States (Grading Logic)
        if self.current_temp > 5.0:
            # Cargo spoiled. End episode immediately.
            done = True
            self.current_location = "Spoiled_En_Route"
            
        elif self.fuel <= 0.0:
            # Out of gas. End episode immediately.
            done = True
            self.fuel = 0.0
            self.current_location = "Stranded_Empty_Fuel"
            
        elif self.dist_dest <= 0.0 and target == "Destination":
            # Successful Delivery!
            done = True
            self.current_location = "Destination"
            # Terminal Bonus: 0.2 base + up to 0.3 depending on fuel efficiency
            # Ensures total possible score across the whole episode is exactly 1.0
            step_reward += 0.2 + 0.3 * (self.fuel / 100.0)
            
        elif self.dist_hub <= 0.0 and target != "Destination":
            # Reached emergency hub
            done = True
            self.current_location = "Repair_Hub"
            # Partial success terminal bonus
            step_reward += 0.1
            
        # Max steps safety net (prevents infinite loops)
        if self._state.step_count >= 20 and not done:
            done = True

        return self._generate_observation(reward=step_reward, done=done)

    def _generate_observation(self, reward: float, done: bool) -> MyEnvObservation:
        """Helper to construct the strict Pydantic observation model."""
        return MyEnvObservation(
            current_location=self.current_location,
            cargo_temp_celsius=float(round(self.current_temp, 2)),
            fuel_level_percent=float(max(0.0, round(self.fuel, 2))),
            ambient_temp_celsius=float(self.ambient_temp),
            distance_to_destination_km=float(round(self.dist_dest, 2)),
            distance_to_emergency_hub_km=float(round(self.dist_hub, 2)),
            cooling_unit_health=float(self.cooling_health),
            done=done,
            reward=reward
        )

    @property
    def state(self) -> State:
        """Get the current OpenEnv internal state."""
        return self._state