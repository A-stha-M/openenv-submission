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
from typing import Dict
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

    MAX_STEPS: int = 20

    TASK_CONFIGS: Dict[str, Dict[str, object]] = {
        "cold_chain_easy": {
            "distance_km": 280.0,
            "hub_distance_km": 120.0,
            "ambient_base_c": 26.0,
            "base_traffic": 1.05,
            "cooling_health": 0.96,
            "deadline_hours": 7,
            "heat_profile": [0.0, 0.8, 1.1, 0.4, -0.2, 0.6, 0.0, -0.4, 0.3, 0.0],
            "traffic_profile": [1.00, 1.05, 1.10, 1.00, 0.95, 1.05, 1.00, 1.00, 0.98, 1.00],
            "repair_cooling_boost": 0.12,
            "repair_fuel_bonus": 8.0,
        },
        "cold_chain_medium": {
            "distance_km": 420.0,
            "hub_distance_km": 140.0,
            "ambient_base_c": 34.0,
            "base_traffic": 1.25,
            "cooling_health": 0.78,
            "deadline_hours": 9,
            "heat_profile": [1.0, 2.0, 3.0, 1.5, 0.0, 2.5, 3.5, 1.0, 0.5, 2.0, 1.0],
            "traffic_profile": [1.05, 1.15, 1.30, 1.20, 1.10, 1.25, 1.35, 1.15, 1.10, 1.20, 1.15],
            "repair_cooling_boost": 0.20,
            "repair_fuel_bonus": 12.0,
        },
        "cold_chain_hard": {
            "distance_km": 640.0,
            "hub_distance_km": 170.0,
            "ambient_base_c": 38.0,
            "base_traffic": 1.55,
            "cooling_health": 0.62,
            "deadline_hours": 11,
            "heat_profile": [2.0, 3.5, 4.0, 5.0, 2.5, 3.0, 5.5, 4.0, 2.0, 3.5, 4.5, 2.5],
            "traffic_profile": [1.20, 1.35, 1.50, 1.60, 1.40, 1.55, 1.70, 1.55, 1.45, 1.50, 1.60, 1.50],
            "repair_cooling_boost": 0.28,
            "repair_fuel_bonus": 16.0,
        },
    }

    SCORE_WEIGHTS: Dict[str, Dict[str, float]] = {
        "cold_chain_easy": {
            "completion": 0.35,
            "cargo": 0.25,
            "fuel": 0.20,
            "discipline": 0.10,
            "schedule": 0.10,
        },
        "cold_chain_medium": {
            "completion": 0.30,
            "cargo": 0.30,
            "fuel": 0.15,
            "discipline": 0.10,
            "schedule": 0.15,
        },
        "cold_chain_hard": {
            "completion": 0.28,
            "cargo": 0.32,
            "fuel": 0.10,
            "discipline": 0.15,
            "schedule": 0.15,
        },
    }

    def __init__(self):
        """Initialize the logistics environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        self.task_id = "cold_chain_easy"
        self.config: Dict[str, object] = self.TASK_CONFIGS[self.task_id]

        self.initial_dist = 0.0
        self.dist_dest = 0.0
        self.dist_hub = 0.0
        self.current_temp = 2.0
        self.ambient_temp = 25.0
        self.fuel = 100.0
        self.base_traffic = 1.0
        self.traffic_multiplier = 1.0
        self.cooling_health = 1.0
        self.current_location = "Origin"
        self.deadline_hours = 8

        self.last_target: str | None = None
        self.route_switch_count = 0
        self.idle_steps = 0
        self.safety_breach_steps = 0
        self.hub_visits = 0
        self.max_temp_seen = 2.0
        self.done = False

    def reset(self, **kwargs) -> MyEnvObservation:
        """
        Reset the environment based on the task.
        Accepts dynamic task injection via kwargs from the HTTP API, 
        or falls back to environment variables.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        # Read the task difficulty injected by evaluator scripts
        self.task_id = kwargs.get("task_name", os.getenv("MY_ENV_TASK", "cold_chain_easy"))

        if self.task_id not in self.TASK_CONFIGS:
            self.task_id = "cold_chain_easy"

        self.config = self.TASK_CONFIGS[self.task_id]

        self.initial_dist = float(self.config["distance_km"])
        self.dist_dest = self.initial_dist
        self.dist_hub = float(self.config["hub_distance_km"])
        self.current_temp = 2.2
        self.fuel = 100.0
        self.current_location = "En_Route"
        self.ambient_temp = float(self.config["ambient_base_c"])
        self.base_traffic = float(self.config["base_traffic"])
        self.traffic_multiplier = self.base_traffic
        self.cooling_health = float(self.config["cooling_health"])
        self.deadline_hours = int(self.config["deadline_hours"])

        self.last_target = None
        self.route_switch_count = 0
        self.idle_steps = 0
        self.safety_breach_steps = 0
        self.hub_visits = 0
        self.max_temp_seen = self.current_temp
        self.done = False

        return self._generate_observation(reward=0.0, done=False)

    def step(self, action: MyEnvAction) -> MyEnvObservation:  # type: ignore[override]
        """
        Execute a 1-hour time step in the logistics simulation.
        """
        if self.done:
            return self._generate_observation(reward=0.0, done=True)

        self._state.step_count += 1
        step_index = max(0, self._state.step_count - 1)

        target = action.target_hub
        speed = float(action.speed_kmh)
        cooling = float(action.cooling_power)

        self._update_step_conditions(step_index)

        # Penalize indecisive target switching, a common failure in dispatch loops.
        switch_penalty = 0.0
        if self.last_target is not None and target != self.last_target:
            self.route_switch_count += 1
            switch_penalty = 0.02 + (0.01 * min(self.route_switch_count, 4))
        self.last_target = target

        # 1. Distance covered in this one-hour step.
        actual_speed = speed / max(1.0, self.traffic_multiplier)
        distance_covered = actual_speed

        step_reward = 0.0
        done = False

        # 2. Update route and dense progress reward with urgency-aware scaling.
        if target == "Destination":
            progress = min(distance_covered, self.dist_dest)
            self.dist_dest -= progress
            self.current_location = "En_Route"
            step_reward += (progress / self.initial_dist) * (0.30 + 0.20 * self._urgency_index())
        else:
            progress = min(distance_covered, self.dist_hub)
            self.dist_hub -= progress
            self.current_location = "Rerouting_Repair_Hub"
            step_reward += (progress / self.initial_dist) * 0.10

        if progress < 1.0:
            self.idle_steps += 1

        step_reward += self._stability_reward(speed=speed, cooling=cooling)

        # 3. Fuel consumption: nonlinear speed + cooling draw + extreme-heat compressor overhead.
        speed_cost = ((speed / 60.0) ** 2) * 1.9
        cooling_cost = (0.9 + self.ambient_temp / 60.0) * cooling * 4.0
        compressor_overhead = max(0.0, cooling - 0.75) * 2.6 + max(0.0, self.ambient_temp - 36.0) * cooling * 0.08
        fuel_burn = speed_cost + cooling_cost + compressor_overhead
        self.fuel -= fuel_burn
        self.fuel = max(0.0, self.fuel)

        # 4. Thermal model + compressor wear from sustained high cooling.
        wear = max(0.0, cooling - 0.82) * 0.02 + max(0.0, self.ambient_temp - 35.0) * 0.002
        self.cooling_health = max(0.35, self.cooling_health - wear)

        insulation_factor = 0.12
        cooling_effect = cooling * self.cooling_health * 4.8
        heat_gained = insulation_factor * (self.ambient_temp - self.current_temp)

        undercooling_drag = max(0.0, 0.55 - cooling) * max(0.0, self.ambient_temp - 30.0) * 0.03
        self.current_temp += (heat_gained - cooling_effect + undercooling_drag)
        self.current_temp = max(-2.0, self.current_temp)
        self.max_temp_seen = max(self.max_temp_seen, self.current_temp)

        # 5. Reaching repair hub applies deterministic maintenance and can be revisited.
        if target == "Repair_Hub" and self.dist_hub <= 0.0:
            self.hub_visits += 1
            self.current_location = "Repair_Hub"
            self.cooling_health = min(1.0, self.cooling_health + float(self.config["repair_cooling_boost"]))
            self.fuel = min(100.0, self.fuel + float(self.config["repair_fuel_bonus"]))
            self.dist_hub = float(self.config["hub_distance_km"])
            step_reward += 0.05
            if self.hub_visits > 1:
                step_reward -= 0.04 * min(self.hub_visits - 1, 3)

        # 6. Safety and behavior penalties.
        safety_penalty = 0.0
        if self.current_temp >= 4.6:
            self.safety_breach_steps += 1
            safety_penalty += 0.08
        if self.fuel <= 12.0:
            safety_penalty += 0.05
        if self.current_temp < 0.8 and cooling > 0.85:
            safety_penalty += 0.03
        if target == "Destination" and progress < 1.0:
            safety_penalty += 0.02

        step_reward -= switch_penalty
        step_reward -= safety_penalty

        # 7. Terminal states and deterministic task grading.
        if self.current_temp > 5.0:
            done = True
            self.current_location = "Spoiled_En_Route"
        elif self.fuel <= 0.0:
            done = True
            self.current_location = "Stranded_Empty_Fuel"
        elif self.dist_dest <= 0.0 and target == "Destination":
            done = True
            self.current_location = "Destination"
            final_score = self._task_score(terminal=True)
            step_reward += 0.20 + 0.20 * final_score

        if self._state.step_count >= self.MAX_STEPS and not done:
            done = True
            self.current_location = "Time_Expired"

        if done and self.current_location != "Destination":
            step_reward = 0.0

        self.done = done
        step_reward = self._clamp(step_reward, 0.0, 1.0)

        return self._generate_observation(reward=step_reward, done=done)

    def _update_step_conditions(self, step_index: int) -> None:
        """Apply deterministic weather and traffic pulses for current step."""
        heat_profile = self.config["heat_profile"]
        traffic_profile = self.config["traffic_profile"]
        assert isinstance(heat_profile, list)
        assert isinstance(traffic_profile, list)

        heat_offset = float(heat_profile[min(step_index, len(heat_profile) - 1)])
        traffic_factor = float(traffic_profile[min(step_index, len(traffic_profile) - 1)])

        self.ambient_temp = float(self.config["ambient_base_c"]) + heat_offset
        self.traffic_multiplier = self.base_traffic * traffic_factor

    def _stability_reward(self, speed: float, cooling: float) -> float:
        """Reward smooth transport operation inside efficient thermal/energy bands."""
        reward = 0.0
        if 1.4 <= self.current_temp <= 4.2:
            reward += 0.05
        if 62.0 <= speed <= 85.0 and 0.45 <= cooling <= 0.75:
            reward += 0.04
        return reward

    def _urgency_index(self) -> float:
        """Estimate urgency from remaining distance, elapsed time, and thermal risk."""
        remaining_ratio = self.dist_dest / max(self.initial_dist, 1.0)
        time_pressure = self._state.step_count / max(float(self.deadline_hours), 1.0)
        thermal_pressure = max(0.0, self.current_temp - 3.5) / 1.5
        urgency = 0.45 * remaining_ratio + 0.35 * time_pressure + 0.20 * thermal_pressure
        return self._clamp(urgency, 0.0, 1.0)

    def _cargo_quality_index(self) -> float:
        """Track cargo quality degradation via worst-case temperature exposure."""
        if self.max_temp_seen >= 5.0:
            return 0.0
        quality = 1.0 - max(0.0, self.max_temp_seen - 2.0) / 3.0
        return self._clamp(quality, 0.0, 1.0)

    def _task_score(self, terminal: bool) -> float:
        """Deterministic task grader producing a score in [0.0, 1.0]."""
        completion = self._clamp(1.0 - (self.dist_dest / max(self.initial_dist, 1.0)), 0.0, 1.0)
        cargo = self._cargo_quality_index()
        fuel_eff = self._clamp(self.fuel / 100.0, 0.0, 1.0)

        discipline = 1.0 - (0.12 * self.route_switch_count) - (0.05 * self.idle_steps) - (0.07 * self.safety_breach_steps)
        discipline = self._clamp(discipline, 0.0, 1.0)

        if self.current_location == "Destination":
            lateness = max(0, self._state.step_count - self.deadline_hours)
            schedule = self._clamp(1.0 - (lateness / max(float(self.deadline_hours), 1.0)), 0.0, 1.0)
        elif terminal:
            schedule = 0.0
        else:
            time_left = max(0, self.deadline_hours - self._state.step_count)
            expected_remaining = self.dist_dest / max(self.initial_dist, 1.0)
            schedule = self._clamp(1.0 - max(0.0, expected_remaining - (time_left / max(float(self.deadline_hours), 1.0))), 0.0, 1.0)

        weights = self.SCORE_WEIGHTS[self.task_id]
        score = (
            weights["completion"] * completion
            + weights["cargo"] * cargo
            + weights["fuel"] * fuel_eff
            + weights["discipline"] * discipline
            + weights["schedule"] * schedule
        )

        if terminal and self.current_location == "Destination":
            strategic_bonus = 0.0
            if self.task_id in {"cold_chain_medium", "cold_chain_hard"} and self.hub_visits == 1:
                strategic_bonus = 0.05
            score += strategic_bonus

        if terminal and self.current_location != "Destination":
            score = min(score, 0.35)

        return self._clamp(score, 0.0, 1.0)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _generate_observation(self, reward: float, done: bool) -> MyEnvObservation:
        """Helper to construct the strict Pydantic observation model."""
        return MyEnvObservation(
            task_name=self.task_id,
            current_location=self.current_location,
            cargo_temp_celsius=float(round(self.current_temp, 2)),
            fuel_level_percent=float(max(0.0, round(self.fuel, 2))),
            ambient_temp_celsius=float(self.ambient_temp),
            distance_to_destination_km=float(round(self.dist_dest, 2)),
            distance_to_emergency_hub_km=float(round(self.dist_hub, 2)),
            cooling_unit_health=float(self.cooling_health),
            hours_elapsed=int(self._state.step_count),
            delivery_deadline_hours=int(self.deadline_hours),
            urgency_index=float(round(self._urgency_index(), 3)),
            cargo_quality_index=float(round(self._cargo_quality_index(), 3)),
            route_switch_count=int(self.route_switch_count),
            task_score=float(round(self._task_score(terminal=done), 3)),
            done=done,
            reward=reward
        )

    @property
    def state(self) -> State:
        """Get the current OpenEnv internal state."""
        return self._state