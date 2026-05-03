"""
Document Intelligence API
==========================
FastAPI server for Malaysian document extraction.

Endpoints:
    POST /extract           — Upload image, get structured JSON
    POST /extract/batch     — Upload multiple images
    GET  /supported-types   — List supported document types
    GET  /health            — Health check

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import time
import logging
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Global extractor
_extractor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load extractor on startup."""
    global _extractor
    from extractors.vlm_extractor import VLMExtractor
    _extractor = VLMExtractor()  # Auto-detect best backend
    logger.info(f"Document extractor ready | backend={_extractor._active_backend}")
    yield


app = FastAPI(
    title="Malaysian Document Intelligence API",
    description=(
        "Extract structured data from Malaysian documents using "
        "Vision-Language Models (Qwen2.5-VL) and OCR.\n\n"
        "**Supported documents:** MyKad, SG NRIC, SSM Certificate, "
        "Invoice, EA Form, Payslip, Bank Statement, Utility Bill, EPF Statement\n\n"
        "**Languages:** Bahasa Malaysia + English\n\n"
        "Built by Aliya Alias | github.com/aliyaalias19"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractionResponse(BaseModel):
    document_type: str
    confidence: float
    extracted_data: dict
    model_used: str
    latency_ms: float
    errors: list[str] = []
    warnings: list[str] = []


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "backend": _extractor._active_backend if _extractor else "not_loaded",
        "supported_types": 9,
    }


@app.get("/supported-types")
async def supported_types():
    from models.document_schemas import DOCUMENT_DESCRIPTIONS
    return {
        "supported_document_types": DOCUMENT_DESCRIPTIONS,
        "total": len(DOCUMENT_DESCRIPTIONS),
        "languages": ["Bahasa Malaysia", "English"],
        "note": "Document type is auto-detected if not specified",
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract_document(
    file: UploadFile = File(..., description="Document image (JPG, PNG, PDF)"),
    doc_type: Optional[str] = Form(None, description="Override document type detection"),
):
    """
    Extract structured data from a document image.

    Upload any Malaysian document and receive structured JSON output.
    Document type is auto-detected if not specified.
    """
    if _extractor is None:
        raise HTTPException(503, "Extractor not initialised")

    # Validate file type
    allowed = {".jpg", ".jpeg", ".png", ".pdf", ".webp", ".bmp", ".tiff"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Use: {allowed}")

    # Save to temp file
    import tempfile
    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = _extractor.extract(
            image_path=tmp_path,
            doc_type=doc_type,
            filename=file.filename,
        )
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return ExtractionResponse(
        document_type=result.document_type,
        confidence=result.confidence,
        extracted_data=result.extracted_data,
        model_used=result.model_used,
        latency_ms=result.latency_ms,
        errors=result.errors,
        warnings=result.warnings,
    )


@app.post("/extract/batch")
async def extract_batch(
    files: list[UploadFile] = File(...),
    doc_type: Optional[str] = Form(None),
):
    """Extract from multiple documents in one request."""
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 files per batch request")

    results = []
    for f in files:
        try:
            single = await extract_document(file=f, doc_type=doc_type)
            results.append({"filename": f.filename, "result": single.model_dump()})
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return {"batch_results": results, "total": len(results)}


@app.get("/")
async def root():
    return {
        "message": "Malaysian Document Intelligence API",
        "docs": "/docs",
        "health": "/health",
        "supported_types": "/supported-types",
        "extract": "POST /extract",
        "author": "Aliya Alias | linkedin.com/in/aliyaalias",
    }
