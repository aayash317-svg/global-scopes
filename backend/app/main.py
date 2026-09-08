"""
FastAPI Main Application Entry Point
Voice Integrity Verification Framework (SIH Problem Statement 26104)
Theme: Blockchain & Cybersecurity
"""

import sys
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, status, UploadFile, File, Form, Depends, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import settings
from backend.app.database.session import init_db, async_engine, get_db
from backend.app.edge_queue import EdgeQueueService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("voice_integrity.app")

edge_queue = EdgeQueueService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan."""
    logger.info("Initializing database tables...")
    try:
        await init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise e
    yield
    logger.info("Shutting down application...")
    await async_engine.dispose()
    logger.info("Database engine disposed.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Voice Integrity Verification Framework (SIH26104) - Real-time Voice Authenticity & Impersonation Defense",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(f"HTTP exception on {request.method} {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "status_code": exc.status_code, "message": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": True, "status_code": 422, "message": "Request validation failed", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": True, "status_code": 500, "message": "Internal server error"},
    )


# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time-Sec"] = f"{process_time:.4f}"
    return response


# Health Check Endpoint
@app.get("/health", tags=["Health"], summary="System health and readiness check")
async def health_check() -> Dict[str, Any]:
    db_status = "healthy"
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database": db_status,
        "timestamp": time.time(),
        "device": settings.DEVICE,
    }


# Include Routers with /api/v1 prefix
from backend.app.api.routes.audio import router as audio_router, audio_stream_websocket
from backend.app.api.routes.detection import router as detection_router, analyze_full_pipeline
from backend.app.api.routes.risk import router as risk_router
from backend.app.api.routes.verification import router as verification_router, enroll_voiceprint
from backend.app.api.routes.privacy import router as privacy_router, get_compliance_info
from backend.app.api.routes.audit import router as audit_router, get_audit_chain, verify_chain_integrity

app.include_router(audio_router, prefix=f"{settings.API_PREFIX}/audio", tags=["Audio"])
app.include_router(detection_router, prefix=f"{settings.API_PREFIX}/detection", tags=["Detection"])
app.include_router(risk_router, prefix=f"{settings.API_PREFIX}/risk", tags=["Risk Engine"])
app.include_router(verification_router, prefix=f"{settings.API_PREFIX}/verification", tags=["Verification"])
app.include_router(privacy_router, prefix=f"{settings.API_PREFIX}/privacy", tags=["Privacy & Compliance"])
app.include_router(audit_router, prefix=f"{settings.API_PREFIX}/audit", tags=["Tamper-Evident Audit"])

# Root-level endpoints matching SIH26104 Master Reference specification
@app.post("/enroll", tags=["Root Reference Endpoints"], summary="Voiceprint enrollment at KYC stage")
async def root_enroll(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    consent_granted: bool = Form(True),
    db: AsyncSession = Depends(get_db),
):
    return await enroll_voiceprint(file=file, user_id=user_id, consent_granted=consent_granted, db=db)


@app.post("/analyze", tags=["Root Reference Endpoints"], summary="Full 5-layer voice analysis pipeline")
async def root_analyze(
    file: UploadFile = File(...),
    caller_id: str = Form("caller_default"),
    scenario: str = Form("routine_support"),
    call_origin: str = Form("VoIP"),
    transaction_value: float = Form(0.0),
    is_known_contact: bool = Form(True),
    fraud_history_score: float = Form(0.0),
    session_id: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    return await analyze_full_pipeline(
        file=file,
        caller_id=caller_id,
        scenario=scenario,
        call_origin=call_origin,
        transaction_value=transaction_value,
        is_known_contact=is_known_contact,
        fraud_history_score=fraud_history_score,
        session_id=session_id,
        db=db,
    )


@app.get("/compliance", tags=["Root Reference Endpoints"], summary="DPDP Act 2023 compliance configuration")
async def root_compliance():
    return await get_compliance_info()


@app.get("/audit/chain", tags=["Root Reference Endpoints"], summary="Recent hash-chain blocks")
async def root_audit_chain(limit: int = 50, db: AsyncSession = Depends(get_db)):
    return await get_audit_chain(limit=limit, db=db)


@app.get("/audit/verify", tags=["Root Reference Endpoints"], summary="End-to-end cryptographic tamper verification")
async def root_audit_verify(db: AsyncSession = Depends(get_db)):
    return await verify_chain_integrity(db=db)


@app.websocket("/stream")
async def root_audio_stream(websocket: WebSocket):
    await audio_stream_websocket(websocket)


# Edge / Offline synchronization endpoints
@app.get("/edge/status", tags=["Edge Resilience"], summary="Edge node status and pending sync queue")
async def get_edge_status():
    return edge_queue.get_status()


@app.post("/edge/sync", tags=["Edge Resilience"], summary="Flush queued events to central audit chain")
async def sync_edge_queue(db: AsyncSession = Depends(get_db)):
    return await edge_queue.sync_events(db)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to Voice Integrity Verification Framework API",
        "docs": "/docs",
        "health": "/health",
        "version": settings.VERSION,
    }
