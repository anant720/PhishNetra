"""
RiskAnalyzer AI - Main FastAPI Application
Production-ready scam detection API
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import time

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.api.routes.analyze import router as analyze_router
from app.api.routes.health import router as health_router

# Setup logging
setup_logging()
logger = get_logger("main")

# Create FastAPI app
app = FastAPI(
    title="PhishNetra",
    description="Advanced AI-powered scam detection and analysis system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middlewares
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Trusted host middleware (for production)
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["yourdomain.com", "api.yourdomain.com"]  # Configure for production
    )

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")

    try:
        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"status={response.status_code} duration={process_time:.3f}s"
        )

        return response

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"error={str(e)} duration={process_time:.3f}s"
        )
        raise

# Include routers
app.include_router(
    analyze_router,
    prefix="/api/v1",
    tags=["analysis"]
)

app.include_router(
    health_router,
    prefix="/api/v1",
    tags=["health"]
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "PhishNetra",
        "version": "1.0.0",
        "description": "Advanced AI-powered scam detection system",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("Starting PhishNetra server")

    # Log configuration
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"API Host: {settings.api_host}:{settings.api_port}")
    logger.info(f"Rate Limit: {settings.rate_limit_requests} requests per {settings.rate_limit_window}s")

    # Initialize models (they will be loaded on first use due to lazy loading)
    logger.info("Models will be initialized on first request")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down PhishNetra server")

    # Cleanup tasks would go here (close connections, save models, etc.)
    logger.info("Cleanup completed")


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug_mode,
        log_level=settings.log_level.lower()
    )