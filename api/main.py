"""FastAPI endpoints for the IrisSecure dashboard and biometric gallery."""

import os
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from api.analytics import REPORT_PATH, SCORES_PATH, load_reference_stats, optimize_threshold
from api.database import AnalysisJob, Identity, SessionLocal, initialize_database


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="IrisSecure API",
    description="Iris biometric gallery, verification, and analytics API",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbeddingPayload(BaseModel):
    embedding: list[float] = Field(min_length=512, max_length=512)


def _similarity(left: list[float], right: list[float]) -> float:
    a, b = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _run_local_job(job_id: str, target_far: Optional[float]) -> None:
    with SessionLocal() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None:
            return
        job.status = "running"
        session.commit()
        try:
            job.result = optimize_threshold(target_far=target_far)
            job.status = "completed"
        except Exception as exc:
            job.error = str(exc)[:500]
            job.status = "failed"
        session.commit()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "irissecure-api"}


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    with SessionLocal() as session:
        identities = session.scalar(select(func.count()).select_from(Identity)) or 0
    stats = load_reference_stats()
    genuine = stats.get("intra_class_similarity", {})
    impostor = stats.get("inter_class_similarity", {})
    evaluation = {}
    if SCORES_PATH.exists() and REPORT_PATH.exists():
        with REPORT_PATH.open(encoding="utf-8") as source:
            evaluation = json.load(source)
        genuine = evaluation.get("genuine_stats", genuine)
        impostor = evaluation.get("impostor_stats", impostor)
    return {
        "identity_count": identities,
        "model_available": (Path(__file__).resolve().parents[1] / "models" / "checkpoints" / "best_model.pth").exists(),
        "dataset_reference": {
            "classes": stats.get("total_classes", 0),
            "embeddings": stats.get("total_embeddings", 0),
            "embedding_dimension": stats.get("embedding_dim", 512),
            "genuine_mean": genuine.get("mean"),
            "impostor_mean": impostor.get("mean"),
        },
        "stats_are_reference": not bool(evaluation),
        "empirical_evaluation_available": SCORES_PATH.exists(),
        "evaluation_split": evaluation.get("split"),
    }


@app.get("/api/identities")
def list_identities() -> list[dict[str, Any]]:
    with SessionLocal() as session:
        rows = session.scalars(select(Identity).order_by(Identity.created_at.desc()).limit(100)).all()
        return [
            {"user_id": row.user_id, "eye": row.eye, "person_id": row.person_id,
             "num_samples": row.num_samples, "created_at": row.created_at.isoformat()}
            for row in rows
        ]


@app.post("/api/identities/{user_id}", status_code=201)
def register_identity(user_id: str, payload: EmbeddingPayload, eye: str = "", person_id: str = "") -> dict[str, Any]:
    vector = np.asarray(payload.embedding, dtype=np.float32)
    magnitude = float(np.linalg.norm(vector))
    if magnitude == 0:
        raise HTTPException(status_code=422, detail="Embedding must have nonzero magnitude")
    vector = (vector / magnitude).tolist()
    with SessionLocal() as session:
        if session.get(Identity, user_id):
            raise HTTPException(status_code=409, detail="Identity is already registered")
        row = Identity(user_id=user_id, eye=eye.upper(), person_id=person_id, embedding=vector)
        session.add(row)
        session.commit()
    return {"user_id": user_id, "status": "registered"}


@app.delete("/api/identities/{user_id}", status_code=204)
def delete_identity(user_id: str) -> None:
    with SessionLocal() as session:
        row = session.get(Identity, user_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        session.delete(row)
        session.commit()


@app.post("/api/identify")
def identify(payload: EmbeddingPayload, threshold: float = 0.75) -> dict[str, Any]:
    query = np.asarray(payload.embedding, dtype=np.float32).tolist()
    with SessionLocal() as session:
        rows = session.scalars(select(Identity)).all()
    if not rows:
        raise HTTPException(status_code=404, detail="The identity gallery is empty")
    best = max(rows, key=lambda row: _similarity(query, row.embedding))
    score = _similarity(query, best.embedding)
    return {"user_id": best.user_id if score >= threshold else None,
            "similarity": score, "threshold": threshold, "matched": score >= threshold}


@app.post("/api/verify/{user_id}")
def verify_identity(user_id: str, payload: EmbeddingPayload, threshold: float = 0.75) -> dict[str, Any]:
    with SessionLocal() as session:
        identity = session.get(Identity, user_id)
        if identity is None:
            raise HTTPException(status_code=404, detail="Identity not found in the gallery")
        score = _similarity(payload.embedding, identity.embedding)
    return {"user_id": user_id, "similarity": score, "threshold": threshold, "verified": score >= threshold}


@app.post("/api/analytics/optimize", status_code=202)
def create_optimization_job(
    background_tasks: BackgroundTasks,
    target_far: Optional[float] = Query(default=None, ge=0.0, le=1.0),
) -> dict[str, str]:
    job_id = str(uuid.uuid4())
    with SessionLocal() as session:
        session.add(AnalysisJob(id=job_id, status="queued"))
        session.commit()
    background_tasks.add_task(_run_local_job, job_id, target_far)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with SessionLocal() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        return {"job_id": job.id, "status": job.status, "result": job.result,
                "error": job.error, "created_at": job.created_at.isoformat()}


@app.get("/api/analytics/threshold")
def threshold_analytics(target_far: Optional[float] = Query(default=None, ge=0.0, le=1.0)) -> dict[str, Any]:
    try:
        return optimize_threshold(target_far=target_far)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/recognition/embedding", include_in_schema=False)
async def extract_embedding(image: UploadFile = File(...)) -> dict[str, Any]:
    """Extract a model embedding when trained weights are installed locally."""
    from io import BytesIO
    from pathlib import Path
    import cv2

    checkpoint = Path(__file__).resolve().parents[1] / "models" / "checkpoints" / "best_model.pth"
    if not checkpoint.exists():
        raise HTTPException(status_code=503, detail="Train the model before image recognition is available")
    from PIL import Image
    import torch
    from torchvision import transforms
    from models.feature_extractor import IrisEmbeddingNet
    from utils.preprocessing import apply_clahe

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(app.state, "model", None)
    if model is None:
        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
        model_config = checkpoint_data.get("config", {}).get("model", {})
        model = IrisEmbeddingNet(backbone=model_config.get("backbone", "resnet50"),
                                 embedding_dim=model_config.get("embedding_dim", 512), pretrained=False)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        app.state.model = model.to(device).eval()
    raw = await image.read()
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise HTTPException(status_code=422, detail="Uploaded file is not a readable image")
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = apply_clahe(frame)
    frame = cv2.resize(frame, (224, 224))
    rgb = np.stack([frame] * 3, axis=-1)
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = app.state.model(tensor, normalize=True).cpu().numpy().flatten().tolist()
    return {"embedding": embedding}
