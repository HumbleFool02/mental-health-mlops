"""
Pydantic schemas for API request/response validation
"""

from typing import Dict

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request model for single prediction"""

    text: str = Field(
        ..., min_length=1, max_length=10000, description="Text to classify"
    )

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()

    class Config:
        schema_extra = {"example": {"text": "I feel anxious and can't sleep at night"}}


class PredictionResponse(BaseModel):
    """Response model for prediction"""

    prediction: str = Field(..., description="Predicted mental health class")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    probabilities: Dict[str, float] = Field(
        ..., description="Probability for each class"
    )
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")
    drift_detected: bool = Field(..., description="Whether drift was detected")

    class Config:
        schema_extra = {
            "example": {
                "prediction": "Anxiety",
                "confidence": 0.87,
                "probabilities": {
                    "Normal": 0.05,
                    "Depression": 0.03,
                    "Suicidal": 0.01,
                    "Anxiety": 0.87,
                    "Bipolar": 0.02,
                    "Stress": 0.02,
                },
                "model_version": "distilbert_exp2",
                "timestamp": "2025-11-10T18:00:00",
                "drift_detected": False,
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    texts: list[str] = Field(
        ..., min_items=1, max_items=100, description="List of texts to classify"
    )

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "I feel anxious",
                    "Everything is going well",
                    "I'm feeling depressed",
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "distilbert_exp2",
                "uptime_seconds": 3600.5,
            }
        }


class ModelInfoResponse(BaseModel):
    """Model information response"""

    model_name: str
    model_version: str
    f1_macro: float
    f1_suicidal: float
    accuracy: float
    training_date: str
    num_classes: int
    classes: list[str]

    class Config:
        schema_extra = {
            "example": {
                "model_name": "DistilBERT",
                "model_version": "distilbert_exp2",
                "f1_macro": 0.8189,
                "f1_suicidal": 0.710,
                "accuracy": 0.7937,
                "training_date": "2025-11-10",
                "num_classes": 6,
                "classes": [
                    "Normal",
                    "Depression",
                    "Suicidal",
                    "Anxiety",
                    "Bipolar",
                    "Stress",
                ],
            }
        }


class DriftStatusResponse(BaseModel):
    """Drift status response"""

    data_drift_detected: bool
    concept_drift_detected: bool
    prediction_drift_detected: bool
    drift_score: float
    last_check: str
    predictions_since_check: int

    class Config:
        schema_extra = {
            "example": {
                "data_drift_detected": False,
                "concept_drift_detected": False,
                "prediction_drift_detected": False,
                "drift_score": 0.015,
                "last_check": "2025-11-10T17:55:00",
                "predictions_since_check": 150,
            }
        }
