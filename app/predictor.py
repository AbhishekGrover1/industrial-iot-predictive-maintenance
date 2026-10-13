"""Model loading and inference."""

from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .schemas import Diagnosis, SensorReadings

MODEL_FILE = "xgboost_predictive_maintenance.joblib"
ENCODER_FILE = "target_encoder.joblib"

# Column names exactly as the model saw them during training.
FEATURES = [
    "Type",
    "Air temperature K",
    "Process temperature K",
    "Rotational speed rpm",
    "Torque Nm",
    "Tool wear min",
]

# Training cast `Type` with astype("category"), which orders the levels
# alphabetically. Declaring them here keeps the integer codes identical
# whether one row or ten thousand are scored at once.
TYPE_LEVELS = ["H", "L", "M"]


class Predictor:
    """Wraps the trained XGBoost classifier and its label encoder."""

    def __init__(self, model: XGBClassifier, label_encoder: LabelEncoder) -> None:
        self._model = model
        self.labels: list[str] = [str(label) for label in label_encoder.classes_]
        self._check_compatibility()

    @classmethod
    def load(cls, model_dir: Path) -> Self:
        model = joblib.load(model_dir / MODEL_FILE)
        encoder = joblib.load(model_dir / ENCODER_FILE)
        return cls(model, encoder)

    def _check_compatibility(self) -> None:
        """Refuse to start if the artefacts and the API contract have drifted apart."""
        features = self._model.get_booster().feature_names
        if features != FEATURES:
            raise RuntimeError(f"Model expects features {features}, service provides {FEATURES}.")
        if self._model.n_classes_ != len(self.labels):
            raise RuntimeError(
                f"Model predicts {self._model.n_classes_} classes, encoder defines {len(self.labels)}."
            )

    def predict(self, readings: SensorReadings) -> Diagnosis:
        frame = pd.DataFrame(
            {
                "Type": pd.Categorical([readings.type], categories=TYPE_LEVELS),
                "Air temperature K": [readings.air_temperature_k],
                "Process temperature K": [readings.process_temperature_k],
                "Rotational speed rpm": [readings.rotational_speed_rpm],
                "Torque Nm": [readings.torque_nm],
                "Tool wear min": [readings.tool_wear_min],
            },
            columns=FEATURES,
        )
        probabilities = self._model.predict_proba(frame)[0]
        ranked = np.argsort(probabilities)[::-1]
        best = int(ranked[0])
        return Diagnosis(
            prediction=self.labels[best],
            confidence_score=round(float(probabilities[best]), 4),
            all_probabilities={self.labels[i]: round(float(probabilities[i]), 4) for i in ranked},
        )
