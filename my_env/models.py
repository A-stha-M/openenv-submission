"""
Data models for the Supply Chain Cold-Storage Rerouting Environment.
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class MyEnvAction(Action):
    """Action for routing and controlling the refrigerated truck."""
    target_hub: Literal["Destination", "Repair_Hub"] = Field(
        ...,
        description="Next route target. Use 'Repair_Hub' to restore cooling health and reduce breakdown risk.",
    )
    cooling_power: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cooling intensity in [0, 1]. Higher values cool faster but increase fuel burn and compressor wear.",
    )
    speed_kmh: float = Field(
        ...,
        ge=40.0,
        le=120.0,
        description="Truck speed in km/h. Higher speed improves progress but burns fuel nonlinearly.",
    )


class MyEnvObservation(Observation):
    """Observation showing the dynamic state of the truck and cargo."""
    task_name: str = Field(..., description="Current task identifier.")
    current_location: str = Field(..., description="Current location of the truck.")
    cargo_temp_celsius: float = Field(..., description="Current temperature of the cargo. Must stay below 5.0 C.")
    fuel_level_percent: float = Field(..., description="Remaining fuel level (0.0 to 100.0).")
    ambient_temp_celsius: float = Field(..., description="Outside weather temperature, which heats the cargo.")
    distance_to_destination_km: float = Field(..., description="Distance to the final delivery point.")
    distance_to_emergency_hub_km: float = Field(..., description="Distance to the nearest safe repair hub.")
    cooling_unit_health: float = Field(..., description="Health of the cooling unit (0.0 to 1.0). Drops during breakdowns.")
    hours_elapsed: int = Field(..., description="Elapsed episode steps in hours.")
    delivery_deadline_hours: int = Field(..., description="Soft deadline that affects task score and urgency shaping.")
    urgency_index: float = Field(..., description="0.0 to 1.0 urgency estimate derived from remaining distance and time budget.")
    cargo_quality_index: float = Field(..., description="0.0 to 1.0 quality proxy derived from worst temperature seen so far.")
    compliance_index: float = Field(..., description="0.0 to 1.0 compliance score based on thermal excursion budget usage.")
    excursion_hours: int = Field(..., description="Number of one-hour windows spent near or above excursion threshold.")
    excursion_budget_hours: int = Field(..., description="Allowed excursion-hour budget before compliance penalties escalate.")
    route_switch_count: int = Field(..., description="How many times the policy switched route target.")
    task_score: float = Field(..., description="Deterministic grader score in [0.0, 1.0]. Final at terminal step.")


class MyEnvReward(BaseModel):
    """Reward model for evaluating agent performance."""
    value: float = Field(..., description="Continuous reward signal. Positive for safe transit, negative for high temperatures, fuel waste, or bad routing.")