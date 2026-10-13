"""Request and response models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MachineType = Literal["L", "M", "H"]


class SensorReadings(BaseModel):
    """One snapshot of machine telemetry.

    Ranges are deliberately wider than the training data but tight enough to
    catch unit mix-ups, such as degrees Celsius sent in place of kelvin.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "M",
                "air_temperature_k": 298.1,
                "process_temperature_k": 308.6,
                "rotational_speed_rpm": 1551,
                "torque_nm": 42.8,
                "tool_wear_min": 0,
            }
        }
    )

    type: MachineType = Field(description="Product quality variant: L, M or H.")
    air_temperature_k: float = Field(ge=250, le=350, description="Air temperature in kelvin.")
    process_temperature_k: float = Field(ge=250, le=400, description="Process temperature in kelvin.")
    rotational_speed_rpm: int = Field(ge=1, le=10_000, description="Rotational speed in revolutions per minute.")
    torque_nm: float = Field(ge=0.1, le=250, description="Torque in newton-metres.")
    tool_wear_min: int = Field(ge=0, le=1_000, description="Accumulated tool wear in minutes.")

    @field_validator("type", mode="before")
    @classmethod
    def normalise_type(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


class Diagnosis(BaseModel):
    """Predicted machine state."""

    prediction: str = Field(description="Most likely machine state.")
    confidence_score: float = Field(ge=0, le=1, description="Probability assigned to the predicted state.")
    all_probabilities: dict[str, float] = Field(description="Probability of every state, highest first.")


class Health(BaseModel):
    """Service status."""

    status: str
    classes: list[str] = Field(description="Machine states the model can predict.")
