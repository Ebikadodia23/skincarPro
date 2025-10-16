from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    age: Optional[int] = None
    skin_type: Optional[str] = None
    skin_concerns: Optional[List[str]] = None
    fitzpatrick_type: Optional[int] = None

class UserCreate(UserBase):
    password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None
    skin_type: Optional[str] = None
    skin_concerns: Optional[List[str]] = None
    fitzpatrick_type: Optional[int] = None

class UserResponse(UserBase):
    id: uuid.UUID
    is_active: bool
    is_verified: bool
    created_at: datetime
    
    class Config:
        orm_mode = True

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: UserResponse

# Analysis schemas
class AnalysisCreate(BaseModel):
    session_id: Optional[str] = None
    image_filename: str
    image_path: str
    image_size: Optional[int] = None
    blur_score: Optional[float] = None

class AnalysisResults(BaseModel):
    predictions: Dict[str, float]
    landmarks: Optional[List[Dict[str, Any]]] = None
    zone_scores: Optional[Dict[str, float]] = None
    heatmap_data: Optional[str] = None
    confidence_score: Optional[float] = None
    skin_tone_analysis: Optional[Dict[str, Any]] = None
    dominant_conditions: Optional[List[str]] = None
    severity_assessment: Optional[Dict[str, str]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    processing_time: Optional[float] = None

class AnalysisResponse(BaseModel):
    id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    session_id: Optional[str] = None
    image_filename: str
    predictions: Dict[str, float]
    landmarks: Optional[List[Dict[str, Any]]] = None
    zone_scores: Optional[Dict[str, float]] = None
    heatmap_data: Optional[str] = None
    confidence_score: Optional[float] = None
    skin_tone_analysis: Optional[Dict[str, Any]] = None
    dominant_conditions: Optional[List[str]] = None
    severity_assessment: Optional[Dict[str, str]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    model_version: Optional[str] = None
    processing_time: Optional[float] = None
    created_at: datetime
    
    class Config:
        orm_mode = True

# Feedback schemas
class FeedbackCreate(BaseModel):
    analysis_id: uuid.UUID
    accuracy_rating: Optional[int] = None
    helpful_rating: Optional[int] = None
    comments: Optional[str] = None
    reported_issues: Optional[List[str]] = None

class FeedbackResponse(BaseModel):
    id: uuid.UUID
    analysis_id: uuid.UUID
    accuracy_rating: Optional[int] = None
    helpful_rating: Optional[int] = None
    comments: Optional[str] = None
    reported_issues: Optional[List[str]] = None
    created_at: datetime
    
    class Config:
        orm_mode = True

# Product recommendation schemas
class ProductRecommendationResponse(BaseModel):
    id: uuid.UUID
    name: str
    brand: Optional[str] = None
    category: str
    description: Optional[str] = None
    key_ingredients: Optional[List[str]] = None
    skin_types: Optional[List[str]] = None
    concerns_addressed: Optional[List[str]] = None
    purchase_url: Optional[str] = None
    image_url: Optional[str] = None
    
    class Config:
        orm_mode = True

# Dashboard/Analytics schemas
class AnalyticsOverview(BaseModel):
    total_analyses: int
    total_users: int
    avg_confidence_score: float
    most_common_conditions: List[Dict[str, Any]]
    recent_analyses_trend: List[Dict[str, Any]]

class UserAnalyticsResponse(BaseModel):
    user_id: uuid.UUID
    total_analyses: int
    avg_confidence_score: float
    condition_history: Dict[str, List[float]]
    last_analysis: Optional[datetime] = None
    improvement_metrics: Optional[Dict[str, float]] = None

# System health schemas  
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    model_status: Dict[str, Any]
    database_status: str
    uptime_seconds: float

class ModelStatus(BaseModel):
    model_loaded: bool
    model_version: str
    supported_conditions: List[str]
    accuracy_metrics: Dict[str, float]
    last_updated: datetime

# File upload schemas
class UploadResponse(BaseModel):
    file_id: str
    user_id: Optional[str] = None
    filename: str
    size: int
    blur_score: float
    quality_score: float
    status: str

# Error schemas
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = datetime.utcnow()

# Pagination schemas
class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    per_page: int
    pages: int
    has_next: bool
    has_prev: bool