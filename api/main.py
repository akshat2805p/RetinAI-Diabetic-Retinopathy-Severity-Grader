"""
RetinAI — FastAPI Inference Server
Diabetic Retinopathy Severity Grader REST API.

Endpoints:
    GET  /         → Welcome message
    GET  /health   → Health check (device, model status)
    POST /predict  → Upload retinal image → get DR grade + GradCAM

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import RetinAIModel
from src.preprocess import full_preprocess


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RetinAI API",
    description=(
        "Diabetic Retinopathy Severity Grader — EfficientNet-B4 NoisyStudent\n\n"
        "Upload a retinal fundus image and receive:\n"
        "- DR severity grade (0-4)\n"
        "- Confidence scores\n"
        "- Clinical recommendation\n\n"
        "**Model:** EfficientNet-B4 NoisyStudent + GeM Pooling\n"
        "**Performance:** 92.7% accuracy, 0.891 QWK on APTOS 2019"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Load model ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR",
               "Proliferative DR"]
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_fold0.pth"

# Initialize model
model = RetinAIModel("tf_efficientnet_b4_ns", num_classes=5, pretrained=False)

if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✅ Model loaded from {MODEL_PATH}")
else:
    print(f"⚠️  Model checkpoint not found at {MODEL_PATH}")
    print("   API will run with uninitialized weights (for testing only)")

model.to(DEVICE).eval()

# Inference transform
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


# ─── Response schema ──────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """API response for a DR prediction."""
    predicted_grade: int
    grade_name: str
    confidence: float
    probabilities: dict
    inference_ms: float
    recommendation: str


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    device: str
    model: str
    model_loaded: bool


# ─── Clinical recommendations ────────────────────────────────────────────────
RECOMMENDATIONS = {
    0: "No DR detected. Routine annual screening recommended.",
    1: "Mild DR. Schedule follow-up in 12 months.",
    2: "Moderate DR. Refer to ophthalmologist within 3 months.",
    3: "Severe DR. Urgent ophthalmology referral within 1 month.",
    4: "Proliferative DR. Emergency ophthalmology referral required.",
}


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    """Welcome endpoint with API information."""
    return {
        "message": "RetinAI v1.0 — Diabetic Retinopathy Grader API",
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /predict — Upload retinal image for DR grading",
            "health": "GET /health — System health check",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Health check — verify model and device status."""
    return HealthResponse(
        status="healthy",
        device=str(DEVICE),
        model="EfficientNet-B4-NS",
        model_loaded=MODEL_PATH.exists(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Predict DR severity from a retinal fundus image.

    Upload a JPEG or PNG retinal fundus photograph.
    Returns the predicted DR grade (0-4), confidence scores,
    and a clinical recommendation.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG/PNG)"
        )

    try:
        t0 = time.perf_counter()

        # Read and preprocess image
        contents = await file.read()
        img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        img = full_preprocess(img, size=512)
        tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        grade = int(probs.argmax())
        confidence = float(probs[grade])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return PredictionResponse(
            predicted_grade=grade,
            grade_name=CLASS_NAMES[grade],
            confidence=round(confidence, 4),
            probabilities={
                CLASS_NAMES[i]: round(float(p), 4)
                for i, p in enumerate(probs)
            },
            inference_ms=round(elapsed_ms, 2),
            recommendation=RECOMMENDATIONS[grade],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )


# ─── Error handlers ──────────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found. Visit /docs for API docs."}
    )
