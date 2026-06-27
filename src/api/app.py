"""
FastAPI application for Mental Health Classification
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from model_loader import ModelLoader
from predictor import MentalHealthPredictor
from schemas import (
    BatchPredictionRequest,
    DriftStatusResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables
model_loader = None
predictor = None
start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global model_loader, predictor, start_time

    # Startup
    logger.info("🚀 Starting Mental Health MLOps API...")
    start_time = time.time()

    try:
        # Load model
        logger.info("Loading DistilBERT model...")
        model_loader = ModelLoader(model_path="models/distilbert_exp2")
        model_loader.load()

        # Initialize predictor
        predictor = MentalHealthPredictor(model_loader)

        logger.info("✅ API ready!")

    except Exception as e:
        logger.error(f"❌ Failed to start API: {e}")
        raise

    yield

    # Shutdown
    logger.info("👋 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Mental Health Classification API",
    description="MLOps pipeline for mental health text classification using DistilBERT",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mental Health Classification API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "health": "/health",
            "model_info": "/model-info",
            "drift_status": "/drift-status",
            "docs": "/docs",
        },
    }


@app.get("/ui")
async def serve_ui():
    """Serve the web UI"""
    from fastapi.responses import FileResponse

    return FileResponse("src/api/static/index.html")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction for a single text

    Args:
        request: PredictionRequest with text to classify

    Returns:
        PredictionResponse with prediction results
    """
    try:
        # Make prediction
        result = predictor.predict(request.text)

        # Check for drift (simple check for now)
        drift_detected = False
        history = predictor.get_prediction_history()
        if len(history) >= 100:
            # Simple drift check: if confidence is consistently low
            recent_confidences = [p["confidence"] for p in history[-100:]]
            avg_confidence = sum(recent_confidences) / len(recent_confidences)
            drift_detected = avg_confidence < 0.6

        # Create response
        response = PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_version="distilbert_exp2",
            timestamp=datetime.now().isoformat(),
            drift_detected=drift_detected,
        )

        return response

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for multiple texts

    Args:
        request: BatchPredictionRequest with list of texts

    Returns:
        List of predictions
    """
    try:
        results = predictor.predict_batch(request.texts)

        responses = []
        for result in results:
            if "error" in result:
                responses.append({"error": result["error"], "prediction": None})
            else:
                responses.append(
                    {
                        "prediction": result["prediction"],
                        "confidence": result["confidence"],
                        "probabilities": result["probabilities"],
                    }
                )

        return {"predictions": responses, "count": len(responses)}

    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint

    Returns:
        HealthResponse with API status
    """
    uptime = time.time() - start_time if start_time else 0

    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        model_version="distilbert_exp2",
        uptime_seconds=uptime,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get model information

    Returns:
        ModelInfoResponse with model details
    """
    return ModelInfoResponse(
        model_name="DistilBERT",
        model_version="distilbert_exp2",
        f1_macro=0.8189,
        f1_suicidal=0.710,
        accuracy=0.7937,
        training_date="2025-11-10",
        num_classes=6,
        classes=model_loader.get_classes(),
    )


@app.get("/drift-status", response_model=DriftStatusResponse)
async def drift_status():
    """
    Get current drift detection status

    Returns:
        DriftStatusResponse with drift metrics
    """
    history = predictor.get_prediction_history()

    # Simple drift detection
    drift_detected = False
    drift_score = 0.0

    if len(history) >= 100:
        recent_confidences = [p["confidence"] for p in history[-100:]]
        avg_confidence = sum(recent_confidences) / len(recent_confidences)
        drift_score = 1.0 - avg_confidence
        drift_detected = avg_confidence < 0.6

    return DriftStatusResponse(
        data_drift_detected=False,  # Would integrate with DataDriftDetector
        concept_drift_detected=drift_detected,
        prediction_drift_detected=False,  # Would integrate with PredictionDriftDetector
        drift_score=drift_score,
        last_check=datetime.now().isoformat(),
        predictions_since_check=len(history),
    )


@app.get("/prediction-distribution")
async def prediction_distribution():
    """Get current prediction distribution"""
    distribution = predictor.get_prediction_distribution()
    return {
        "distribution": distribution,
        "total_predictions": len(predictor.get_prediction_history()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
