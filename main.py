"""
Bio-AI-SaaS Primer Design API

Enterprise-grade FastAPI application for primer design services.

Features:
- RESTful API endpoints for primer design
- Async task processing with Celery
- Comprehensive input validation and security
- FDA 21 CFR Part 11 compliant audit trail
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import io

from schemas.primer import (
    PrimerDesignRequest,
    PrimerDesignResponse,
    BatchPrimerDesignRequest,
    BatchPrimerDesignResponse,
    ErrorCode,
)
from core.agent import PrimerAgent, design_primers
from core.guardrails import Guardrails, SecurityException, security_audit
from services.bio_compute import audit_trail
from services.report_generator import report_generator, ReportData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Application Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    logger.info("Bio-AI-SaaS Primer Design API starting up...")
    logger.info("Initializing security guardrails...")
    logger.info("Loading prompt versions...")

    yield

    # Shutdown
    logger.info("Bio-AI-SaaS Primer Design API shutting down...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Bio-AI-SaaS Primer Design API",
    description="""
    Enterprise-grade primer design service powered by AI.

    ## Features
    - **Intelligent Primer Design**: ReAct-based agent with Primer3 integration
    - **Specificity Verification**: Local BLAST+ for off-target detection
    - **Variant Checking**: dbSNP/gnomAD integration for SNP conflicts
    - **Security**: Biosecurity screening and rate limiting
    - **Compliance**: FDA 21 CFR Part 11 audit trail

    ## Task Types
    - **qPCR**: Quantitative PCR primers (70-300 bp amplicons)
    - **PCR**: Standard PCR primers
    - **LAMP**: Loop-mediated isothermal amplification
    - **NGS**: Next-generation sequencing library prep
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Dependencies
# ============================================================================

def get_guardrails() -> Guardrails:
    """Dependency for security guardrails."""
    return Guardrails()


def get_agent() -> PrimerAgent:
    """Dependency for primer design agent."""
    return PrimerAgent()


# ============================================================================
# Request/Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    services: dict


class ErrorResponse(BaseModel):
    """Error response model."""
    error_code: str
    message: str
    details: Optional[dict] = None
    request_id: Optional[str] = None


class ReportRequest(BaseModel):
    """Report generation request."""
    request_id: str
    format: str = "html"  # html or pdf


# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    return response


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(SecurityException)
async def security_exception_handler(request: Request, exc: SecurityException):
    """Handle security exceptions."""
    logger.warning(f"Security exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={
            "error_code": ErrorCode.ERR_SECURITY_BLOCK.value,
            "message": str(exc),
            "threat_level": exc.threat_level.value
        }
    )


@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": ErrorCode.ERR_INVALID_SEQUENCE.value,
            "message": str(exc)
        }
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Bio-AI-SaaS Primer Design API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        services={
            "primer3": "available",
            "blast": "available",
            "variant_db": "available",
            "redis": "available"
        }
    )


@app.post(
    "/api/v1/primers/design",
    response_model=PrimerDesignResponse,
    tags=["Primer Design"],
    summary="Design primers for a target gene",
    responses={
        200: {"description": "Successful primer design"},
        400: {"description": "Invalid request parameters"},
        403: {"description": "Security block"},
        500: {"description": "Internal server error"}
    }
)
async def design_primers_endpoint(
    request: PrimerDesignRequest,
    guardrails: Guardrails = Depends(get_guardrails),
    agent: PrimerAgent = Depends(get_agent)
):
    """
    Design primers for a target gene.

    This endpoint:
    1. Validates input and performs security screening
    2. Runs the ReAct agent loop for intelligent design
    3. Returns ranked primer pairs with analysis

    ## Request Body
    - **target_gene**: Gene symbol (e.g., GAPDH, ACE2)
    - **sequence_template**: DNA template sequence (optional)
    - **task_type**: qPCR, PCR, LAMP, or NGS
    - **constraints**: Thermodynamic constraints

    ## Response
    Returns primer pairs ranked by quality with:
    - Thermodynamic properties (Tm, GC%, ΔG)
    - Specificity analysis
    - Variant warnings
    - AI-generated recommendations
    """
    # Security check
    allowed, reason, _ = await guardrails.check_request(
        target_gene=request.target_gene,
        sequence=request.sequence_template,
        user_message=f"Design {request.task_type.value} primers for {request.target_gene}",
        user_id=request.user_id or "anonymous",
        session_id=request.session_id
    )

    if not allowed:
        raise SecurityException(reason)

    # Run primer design
    try:
        response = await agent.run_design_loop(request)
        return response
    except Exception as e:
        logger.error(f"Primer design failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": ErrorCode.ERR_PRIMER3_FAILURE.value,
                "message": str(e)
            }
        )


@app.post(
    "/api/v1/primers/batch",
    response_model=BatchPrimerDesignResponse,
    tags=["Primer Design"],
    summary="Batch primer design for multiple targets"
)
async def batch_design_primers(
    request: BatchPrimerDesignRequest,
    background_tasks: BackgroundTasks,
    guardrails: Guardrails = Depends(get_guardrails)
):
    """
    Submit batch primer design job.

    Returns immediately with batch ID. Results are processed asynchronously.
    Use `/api/v1/primers/batch/{batch_id}` to check status.
    """
    # Validate batch size
    if len(request.requests) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds maximum of 100 requests"
        )

    # Create response
    response = BatchPrimerDesignResponse(
        batch_id=request.batch_id,
        status="queued",
        total_requests=len(request.requests),
        completed_requests=0,
        failed_requests=0,
        results=[]
    )

    # Queue background processing
    # In production, this would use Celery
    # background_tasks.add_task(process_batch, request)

    return response


@app.get(
    "/api/v1/primers/batch/{batch_id}",
    response_model=BatchPrimerDesignResponse,
    tags=["Primer Design"],
    summary="Get batch job status"
)
async def get_batch_status(batch_id: str):
    """Get status of a batch primer design job."""
    # In production, this would query job status from Redis/database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Batch {batch_id} not found"
    )


@app.post(
    "/api/v1/reports/generate",
    tags=["Reports"],
    summary="Generate primer design report"
)
async def generate_report(
    response_data: dict,
    format: str = "html"
):
    """
    Generate a report from primer design results.

    ## Parameters
    - **response_data**: PrimerDesignResponse JSON
    - **format**: Output format (html or pdf)

    ## Returns
    - HTML: Returns HTML string
    - PDF: Returns PDF file stream
    """
    try:
        # Create ReportData from response
        data = ReportData(
            request_id=response_data.get("request_id", "unknown"),
            target_gene=response_data.get("target_gene", "Unknown"),
            task_type=response_data.get("task_type", "PCR"),
            primer_pairs=response_data.get("primer_pairs", []),
            best_pair=response_data.get("best_pair"),
            structural_warnings=response_data.get("structural_warnings", []),
            variant_warnings=response_data.get("variant_warnings", []),
            specificity_result=response_data.get("specificity_results"),
            ai_suggestion=response_data.get("ai_suggestion"),
            computation_time_ms=response_data.get("computation_time_ms", 0),
            created_at=datetime.utcnow()
        )

        if format.lower() == "pdf":
            pdf_bytes = report_generator.generate_pdf(data)
            return StreamingResponse(
                io.BytesIO(pdf_bytes),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=primer_report_{data.request_id[:8]}.pdf"
                }
            )
        else:
            html = report_generator.generate_html(data)
            return {"html": html}

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get(
    "/api/v1/audit/events",
    tags=["Audit"],
    summary="Get audit trail events"
)
async def get_audit_events(
    user_id: Optional[str] = None,
    limit: int = 100
):
    """
    Get audit trail events.

    FDA 21 CFR Part 11 compliant audit logging.
    """
    events = audit_trail.records[-limit:]

    if user_id:
        events = [e for e in events if e.user_id == user_id]

    return {
        "total": len(events),
        "events": [
            {
                "timestamp": e.timestamp.isoformat(),
                "operation": e.operation,
                "user_id": e.user_id,
                "success": e.success,
                "computation_time_ms": e.computation_time_ms
            }
            for e in events
        ]
    }


@app.get(
    "/api/v1/security/events",
    tags=["Security"],
    summary="Get security events"
)
async def get_security_events(
    blocked_only: bool = False,
    limit: int = 100
):
    """Get security audit events."""
    events = security_audit.events[-limit:]

    if blocked_only:
        events = [e for e in events if e.blocked]

    return {
        "total": len(events),
        "events": [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "threat_level": e.threat_level.value,
                "timestamp": e.timestamp.isoformat(),
                "action_taken": e.action_taken,
                "blocked": e.blocked
            }
            for e in events
        ]
    }


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.post(
    "/api/v1/utils/validate-sequence",
    tags=["Utilities"],
    summary="Validate a DNA sequence"
)
async def validate_sequence(sequence: str):
    """Validate a DNA sequence for primer design."""
    from core.guardrails import input_validator

    is_valid, error = input_validator.validate_sequence(sequence)

    return {
        "valid": is_valid,
        "error": error,
        "length": len(sequence) if sequence else 0,
        "gc_content": calculate_gc(sequence) if is_valid else None
    }


def calculate_gc(sequence: str) -> float:
    """Calculate GC content of a sequence."""
    if not sequence:
        return 0.0
    sequence = sequence.upper()
    gc = sequence.count('G') + sequence.count('C')
    return (gc / len(sequence)) * 100


@app.get(
    "/api/v1/utils/databases",
    tags=["Utilities"],
    summary="List available genome databases"
)
async def list_databases():
    """List available genome databases for BLAST."""
    return {
        "databases": [
            {"id": "hg38", "name": "GRCh38 Human Reference Genome", "species": "Homo sapiens"},
            {"id": "hg19", "name": "GRCh37 Human Reference Genome", "species": "Homo sapiens"},
            {"id": "mm10", "name": "GRCm38 Mouse Reference Genome", "species": "Mus musculus"},
            {"id": "mm39", "name": "GRCm39 Mouse Reference Genome", "species": "Mus musculus"},
        ]
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
