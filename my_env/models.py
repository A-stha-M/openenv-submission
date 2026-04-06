"""
Data models for the Supply Chain Cold-Storage Rerouting Environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field

class MyEnvAction(Action):
    """Action for routing and controlling the refrigerated truck."""
    target_hub: str = Field(..., description="The ID of the next city/hub to route the truck to (e.g., 'Destination', 'Repair_Hub').")
    cooling_power: float = Field(..., ge=0.0, le=1.0, description="Cooling unit power level (0.0 to 1.0). High power cools cargo faster but consumes more fuel.")
    speed_kmh: float = Field(..., ge=40.0, le=120.0, description="Speed of the truck. Faster speed means less travel time but exponentially higher fuel burn.")

class MyEnvObservation(Observation):
    """Observation showing the dynamic state of the truck and cargo."""
    current_location: str = Field(..., description="Current location of the truck.")
    cargo_temp_celsius: float = Field(..., description="Current temperature of the cargo. Must stay below 5.0 C.")
    fuel_level_percent: float = Field(..., description="Remaining fuel level (0.0 to 100.0).")
    ambient_temp_celsius: float = Field(..., description="Outside weather temperature, which heats the cargo.")
    distance_to_destination_km: float = Field(..., description="Distance to the final delivery point.")
    distance_to_emergency_hub_km: float = Field(..., description="Distance to the nearest safe repair hub.")
    cooling_unit_health: float = Field(..., description="Health of the cooling unit (0.0 to 1.0). Drops during breakdowns.")

class MyEnvReward(BaseModel):
    """Reward model for evaluating agent performance."""
    value: float = Field(..., description="Continuous reward signal. Positive for safe transit, negative for high temperatures, fuel waste, or bad routing.")