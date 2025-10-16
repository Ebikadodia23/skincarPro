from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import os
import json
import uuid
import time
import shutil
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import hashlib
from PIL import Image
import io
import numpy as np
import logging
from pathlib import Path

# Import our modules
from database import get_db, init_db, database_health_check
from models import User, SkinAnalysis, AnalysisFeedback, ProductRecommendation, SystemMetrics
from schemas import *
from auth import (
    authenticate_user, create_access_token, get_current_user, 
    get_current_user_optional, get_password_hash, cleanup_expired_sessions,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from computer_vision import SkinAnalysisCV

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SkinCare Pro API",
    description="AI-Powered Skin Analysis Platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage configuration
UPLOAD_DIR = Path("uploads")
MODELS_DIR = Path("../models")
STATIC_DIR = Path("static")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize ML service
cv_service = SkinAnalysisCV()
logger.info("Initialized AI skin analysis service")

# Application startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and perform startup tasks."""
    logger.info("Starting SkinCare Pro API...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Cleanup expired sessions
    try:
        from database import SessionLocal
        db = SessionLocal()
        cleaned = cleanup_expired_sessions(db)
        db.close()
        logger.info(f"Cleaned up {cleaned} expired sessions")
    except Exception as e:
        logger.error(f"Failed to cleanup sessions: {e}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    start_time = time.time()
    
    # Check database
    db_status = database_health_check()
    
    # Check model status
    model_status = {
        "model_loaded": hasattr(cv_service, 'model') and cv_service.model is not None,
        "model_version": "1.0.0",
        "supported_conditions": ["acne", "hyperpigmentation", "redness", "dehydration", "pore_size", "fine_lines"]
    }
    
    processing_time = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if db_status["status"] == "healthy" else "degraded",
        timestamp=datetime.utcnow(),
        version="2.0.0",
        model_status=model_status,
        database_status=db_status["status"],
        uptime_seconds=processing_time
    )

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint with basic information."""
    return {
        "message": "SkinCare Pro API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Authentication endpoints
@app.post("/api/v1/auth/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    
    # Check if user already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
    
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        email=user_data.email,
        username=user_data.username,
        full_name=user_data.full_name,
        hashed_password=hashed_password,
        age=user_data.age,
        skin_type=user_data.skin_type,
        skin_concerns=user_data.skin_concerns,
        fitzpatrick_type=user_data.fitzpatrick_type
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(db_user.id)}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.from_orm(db_user)
    )

@app.post("/api/v1/auth/login", response_model=Token)
async def login_user(login_data: UserLogin, db: Session = Depends(get_db)):
    """Authenticate and login a user."""
    
    user = authenticate_user(db, login_data.email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.from_orm(user)
    )

# User endpoints
@app.get("/api/v1/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse.from_orm(current_user)

@app.put("/api/v1/users/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user profile."""
    
    for field, value in user_update.dict(exclude_unset=True).items():
        setattr(current_user, field, value)
    
    current_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(current_user)
    
    return UserResponse.from_orm(current_user)

# Image upload and analysis endpoints
@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload and preprocess image for analysis."""
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    try:
        # Read file data
        image_data = await file.read()
        
        # Generate unique ID
        file_id = str(uuid.uuid4())
        
        # Quality checks
        blur_score = calculate_image_blur(image_data)
        quality_score = assess_image_quality(image_data)
        
        if blur_score < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image too blurry (score: {blur_score:.1f}). Please capture a clearer image."
            )
        
        if quality_score < 0.5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image quality too low (score: {quality_score:.1f}). Please ensure good lighting."
            )
        
        # Process and save image
        clean_image_data = strip_exif_and_resize(image_data)
        
        # Save processed image
        filename = f"{file_id}.jpg"
        filepath = UPLOAD_DIR / filename
        
        with open(filepath, 'wb') as f:
            f.write(clean_image_data)
        
        return UploadResponse(
            file_id=file_id,
            filename=filename,
            size=len(clean_image_data),
            blur_score=blur_score,
            quality_score=quality_score,
            status="uploaded"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_skin_image(
    file_id: str,
    background_tasks: BackgroundTasks,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Run comprehensive skin analysis on uploaded image."""
    
    start_time = time.time()
    
    try:
        # Find uploaded image
        filepath = UPLOAD_DIR / f"{file_id}.jpg"
        if not filepath.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found"
            )
        
        # Read image data
        with open(filepath, 'rb') as f:
            image_data = f.read()
        
        # Run AI analysis
        logger.info(f"Starting analysis for file_id: {file_id}")
        results = cv_service.analyze_image(image_data)
        
        processing_time = time.time() - start_time
        
        # Generate recommendations
        recommendations = generate_product_recommendations(
            results.get("predictions", {}), 
            current_user.skin_type if current_user else None,
            db
        )
        
        # Create analysis record
        analysis = SkinAnalysis(
            user_id=current_user.id if current_user else None,
            session_id=file_id if not current_user else None,
            image_filename=f"{file_id}.jpg",
            image_path=str(filepath),
            image_size=len(image_data),
            predictions=results.get("predictions", {}),
            landmarks=results.get("landmarks", []),
            zone_scores=results.get("zone_scores", {}),
            heatmap_data=results.get("heatmap_b64", ""),
            confidence_score=results.get("confidence_score", 0.0),
            skin_tone_analysis=results.get("skin_tone", {}),
            dominant_conditions=get_dominant_conditions(results.get("predictions", {})),
            severity_assessment=assess_condition_severity(results.get("predictions", {})),
            recommendations=recommendations,
            model_version="1.0.0",
            processing_time=processing_time
        )
        
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        
        # Record metrics in background
        background_tasks.add_task(record_analysis_metrics, analysis.id, processing_time)
        
        logger.info(f"Analysis completed for file_id: {file_id} in {processing_time:.2f}s")
        
        return AnalysisResponse.from_orm(analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

# Analysis history endpoints
@app.get("/api/v1/analyses", response_model=List[AnalysisResponse])
async def get_user_analyses(
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's analysis history."""
    
    analyses = db.query(SkinAnalysis)\
        .filter(SkinAnalysis.user_id == current_user.id)\
        .order_by(desc(SkinAnalysis.created_at))\
        .offset(offset)\
        .limit(limit)\
        .all()
    
    return [AnalysisResponse.from_orm(analysis) for analysis in analyses]

@app.get("/api/v1/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_detail(
    analysis_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed analysis results."""
    
    analysis = db.query(SkinAnalysis)\
        .filter(SkinAnalysis.id == analysis_id)\
        .filter(SkinAnalysis.user_id == current_user.id)\
        .first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    return AnalysisResponse.from_orm(analysis)

# Feedback endpoints
@app.post("/api/v1/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    feedback_data: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit feedback for an analysis."""
    
    # Verify analysis exists and belongs to user
    analysis = db.query(SkinAnalysis)\
        .filter(SkinAnalysis.id == feedback_data.analysis_id)\
        .filter(SkinAnalysis.user_id == current_user.id)\
        .first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    # Create feedback
    feedback = AnalysisFeedback(
        analysis_id=feedback_data.analysis_id,
        accuracy_rating=feedback_data.accuracy_rating,
        helpful_rating=feedback_data.helpful_rating,
        comments=feedback_data.comments,
        reported_issues=feedback_data.reported_issues
    )
    
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    
    return FeedbackResponse.from_orm(feedback)

# Product recommendation endpoints
@app.get("/api/v1/recommendations", response_model=List[ProductRecommendationResponse])
async def get_product_recommendations(
    analysis_id: Optional[uuid.UUID] = None,
    skin_type: Optional[str] = None,
    conditions: Optional[str] = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Get personalized product recommendations."""
    
    if analysis_id and current_user:
        # Get recommendations for specific analysis
        analysis = db.query(SkinAnalysis)\
            .filter(SkinAnalysis.id == analysis_id)\
            .filter(SkinAnalysis.user_id == current_user.id)\
            .first()
        
        if analysis and analysis.recommendations:
            product_ids = [rec.get("product_id") for rec in analysis.recommendations if rec.get("product_id")]
            if product_ids:
                products = db.query(ProductRecommendation)\
                    .filter(ProductRecommendation.id.in_(product_ids))\
                    .filter(ProductRecommendation.is_active == True)\
                    .all()
                return [ProductRecommendationResponse.from_orm(product) for product in products]
    
    # Get general recommendations
    query = db.query(ProductRecommendation).filter(ProductRecommendation.is_active == True)
    
    if skin_type:
        query = query.filter(ProductRecommendation.skin_types.contains([skin_type]))
    
    if conditions:
        condition_list = conditions.split(",")
        for condition in condition_list:
            query = query.filter(ProductRecommendation.concerns_addressed.contains([condition.strip()]))
    
    products = query.limit(10).all()
    return [ProductRecommendationResponse.from_orm(product) for product in products]

# Analytics endpoints
@app.get("/api/v1/analytics/overview", response_model=AnalyticsOverview)
async def get_analytics_overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user analytics overview."""
    
    # Get user's analysis count
    total_analyses = db.query(SkinAnalysis)\
        .filter(SkinAnalysis.user_id == current_user.id)\
        .count()
    
    # Get average confidence score
    avg_confidence = db.query(func.avg(SkinAnalysis.confidence_score))\
        .filter(SkinAnalysis.user_id == current_user.id)\
        .scalar() or 0.0
    
    # Get most common conditions
    user_analyses = db.query(SkinAnalysis)\
        .filter(SkinAnalysis.user_id == current_user.id)\
        .all()
    
    condition_counts = {}
    for analysis in user_analyses:
        if analysis.predictions:
            for condition, score in analysis.predictions.items():
                if score > 0.3:  # Only count significant conditions
                    condition_counts[condition] = condition_counts.get(condition, 0) + 1
    
    most_common = sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    most_common_formatted = [{"condition": cond, "count": count} for cond, count in most_common]
    
    # Get recent trend (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_analyses = db.query(SkinAnalysis)\
        .filter(SkinAnalysis.user_id == current_user.id)\
        .filter(SkinAnalysis.created_at >= thirty_days_ago)\
        .order_by(SkinAnalysis.created_at)\
        .all()
    
    trend_data = []
    for analysis in recent_analyses[-10:]:  # Last 10 analyses
        trend_data.append({
            "date": analysis.created_at.isoformat(),
            "confidence": analysis.confidence_score or 0.0,
            "conditions_detected": len([c for c, s in (analysis.predictions or {}).items() if s > 0.3])
        })
    
    return AnalyticsOverview(
        total_analyses=total_analyses,
        total_users=1,  # Current user
        avg_confidence_score=avg_confidence,
        most_common_conditions=most_common_formatted,
        recent_analyses_trend=trend_data
    )

# Utility functions
def calculate_image_blur(image_data: bytes) -> float:
    """Calculate blur metric for image quality check."""
    try:
        image = Image.open(io.BytesIO(image_data)).convert('L')
        img_array = np.array(image)
        # Laplacian variance for blur detection
        laplacian_var = np.var(np.gradient(img_array))
        return float(laplacian_var)
    except Exception:
        return 0.0

def assess_image_quality(image_data: bytes) -> float:
    """Assess overall image quality."""
    try:
        image = Image.open(io.BytesIO(image_data))
        
        # Basic quality metrics
        width, height = image.size
        aspect_ratio = width / height
        size_score = min(1.0, (width * height) / (640 * 480))  # Prefer at least VGA
        aspect_score = 1.0 - abs(aspect_ratio - 1.0)  # Prefer square-ish images
        
        return (size_score + aspect_score) / 2.0
    except Exception:
        return 0.0

def strip_exif_and_resize(image_data: bytes, max_size: int = 1024) -> bytes:
    """Remove EXIF data and resize image for privacy and efficiency."""
    try:
        image = Image.open(io.BytesIO(image_data))
        
        # Remove EXIF by creating new image
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # Resize if too large
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.LANCZOS)
        
        # Save to bytes
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='JPEG', quality=85, optimize=True)
        return output_buffer.getvalue()
    except Exception:
        return image_data

def get_dominant_conditions(predictions: Dict[str, float]) -> List[str]:
    """Get list of dominant conditions from predictions."""
    if not predictions:
        return []
    
    # Filter conditions with significant scores
    significant = [(condition, score) for condition, score in predictions.items() if score > 0.3]
    significant.sort(key=lambda x: x[1], reverse=True)
    
    return [condition for condition, _ in significant[:3]]

def assess_condition_severity(predictions: Dict[str, float]) -> Dict[str, str]:
    """Assess severity level for each condition."""
    severity_map = {}
    
    for condition, score in predictions.items():
        if score >= 0.7:
            severity = "High"
        elif score >= 0.5:
            severity = "Moderate"
        elif score >= 0.3:
            severity = "Mild"
        else:
            severity = "Low"
        
        severity_map[condition] = severity
    
    return severity_map

def generate_product_recommendations(
    predictions: Dict[str, float], 
    user_skin_type: Optional[str],
    db: Session
) -> List[Dict[str, Any]]:
    """Generate product recommendations based on analysis results."""
    
    recommendations = []
    
    # Get products for detected conditions
    for condition, score in predictions.items():
        if score > 0.3:  # Only recommend for significant conditions
            products = db.query(ProductRecommendation)\
                .filter(ProductRecommendation.concerns_addressed.contains([condition]))\
                .filter(ProductRecommendation.is_active == True)
            
            # Filter by skin type if available
            if user_skin_type:
                products = products.filter(
                    ProductRecommendation.skin_types.contains([user_skin_type])
                )
            
            # Filter by severity range
            products = products.filter(
                ProductRecommendation.severity_range["min"].astext.cast(float) <= score
            ).filter(
                ProductRecommendation.severity_range["max"].astext.cast(float) >= score
            )
            
            product = products.first()
            if product:
                recommendations.append({
                    "product_id": str(product.id),
                    "condition": condition,
                    "severity_score": score,
                    "recommendation_strength": min(1.0, score * 1.5),
                    "reason": f"Recommended for {condition} (severity: {score:.1f})"
                })
    
    return recommendations[:5]  # Limit to top 5 recommendations

async def record_analysis_metrics(analysis_id: uuid.UUID, processing_time: float):
    """Record system metrics for analysis performance."""
    try:
        from database import SessionLocal
        db = SessionLocal()
        
        # Record processing time metric
        metric = SystemMetrics(
            metric_name="analysis_processing_time",
            metric_value=processing_time,
            metric_metadata={"analysis_id": str(analysis_id)}
        )
        db.add(metric)
        
        # Record analysis count
        count_metric = SystemMetrics(
            metric_name="analysis_count",
            metric_value=1.0,
            metric_metadata={"type": "increment"}
        )
        db.add(count_metric)
        
        db.commit()
        db.close()
    except Exception as e:
        logger.error(f"Failed to record metrics: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )