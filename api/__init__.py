"""
RetinAI — FastAPI Inference Server Package

Exposes the FastAPI application for Diabetic Retinopathy grading.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /         → Welcome message + API info
    GET  /health   → Health check (device, model status)
    POST /predict  → Upload retinal image → DR grade + confidence
"""

from api.main import app

__all__ = ["app"]
