from sqlalchemy import Column, String, DateTime, Float, Text, Boolean, Integer, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # User profile information
    age = Column(Integer, nullable=True)
    skin_type = Column(String, nullable=True)  # oily, dry, combination, sensitive, normal
    skin_concerns = Column(JSON, nullable=True)  # array of concerns
    fitzpatrick_type = Column(Integer, nullable=True)  # 1-6
    
    # Relationships
    analyses = relationship("SkinAnalysis", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_token = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    user = relationship("User", back_populates="sessions")

class SkinAnalysis(Base):
    __tablename__ = "skin_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    session_id = Column(String, nullable=True)  # For anonymous users
    
    # Image information
    image_filename = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    image_size = Column(Integer, nullable=True)
    image_quality_score = Column(Float, nullable=True)
    blur_score = Column(Float, nullable=True)
    
    # Analysis results
    predictions = Column(JSON, nullable=False)  # {condition: score}
    landmarks = Column(JSON, nullable=True)  # facial landmarks
    zone_scores = Column(JSON, nullable=True)  # zone-based analysis
    heatmap_data = Column(Text, nullable=True)  # base64 heatmap
    confidence_score = Column(Float, nullable=True)
    
    # Skin analysis details
    skin_tone_analysis = Column(JSON, nullable=True)
    dominant_conditions = Column(JSON, nullable=True)
    severity_assessment = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    
    # Metadata
    model_version = Column(String, nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="analyses")
    feedback = relationship("AnalysisFeedback", back_populates="analysis", cascade="all, delete-orphan")

class AnalysisFeedback(Base):
    __tablename__ = "analysis_feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("skin_analyses.id"), nullable=False)
    
    # User feedback
    accuracy_rating = Column(Integer, nullable=True)  # 1-5 stars
    helpful_rating = Column(Integer, nullable=True)  # 1-5 stars
    comments = Column(Text, nullable=True)
    reported_issues = Column(JSON, nullable=True)
    
    # Professional feedback (if clinician reviewed)
    clinician_id = Column(String, nullable=True)
    professional_assessment = Column(JSON, nullable=True)
    corrections = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analysis = relationship("SkinAnalysis", back_populates="feedback")

class ProductRecommendation(Base):
    __tablename__ = "product_recommendations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    brand = Column(String, nullable=True)
    category = Column(String, nullable=False)  # cleanser, moisturizer, treatment, etc.
    
    # Product details
    description = Column(Text, nullable=True)
    key_ingredients = Column(JSON, nullable=True)
    skin_types = Column(JSON, nullable=True)  # suitable for which skin types
    concerns_addressed = Column(JSON, nullable=True)  # which conditions it helps
    
    # Recommendation logic
    severity_range = Column(JSON, nullable=True)  # min/max severity scores
    contraindications = Column(JSON, nullable=True)
    
    # External links
    purchase_url = Column(String, nullable=True)
    image_url = Column(String, nullable=True)
    
    # Metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SystemMetrics(Base):
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_metadata = Column(JSON, nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    
    # Common metrics: analysis_count, avg_processing_time, model_accuracy, etc.