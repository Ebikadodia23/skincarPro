from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
from models import Base

# Database URL configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./skincare_pro.db"
)

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL debugging
    )
else:
    # For PostgreSQL in production
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=0,
        pool_recycle=300,
        pool_pre_ping=True,
        echo=False
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)

def drop_tables():
    """Drop all database tables."""
    Base.metadata.drop_all(bind=engine)

def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize the database with tables and default data."""
    create_tables()
    
    # Add default product recommendations
    from models import ProductRecommendation
    db = SessionLocal()
    
    # Check if we already have product recommendations
    if db.query(ProductRecommendation).count() == 0:
        default_products = [
            ProductRecommendation(
                name="Gentle Foam Cleanser",
                brand="CeraVe",
                category="cleanser",
                description="A gentle, non-comedogenic cleanser suitable for all skin types",
                key_ingredients=["ceramides", "hyaluronic acid", "niacinamide"],
                skin_types=["normal", "dry", "combination", "sensitive"],
                concerns_addressed=["dehydration", "redness"],
                severity_range={"min": 0.0, "max": 1.0}
            ),
            ProductRecommendation(
                name="Salicylic Acid Cleanser",
                brand="The Ordinary",
                category="treatment",
                description="BHA cleanser for acne-prone and oily skin",
                key_ingredients=["salicylic acid", "zinc"],
                skin_types=["oily", "combination"],
                concerns_addressed=["acne", "pore_size"],
                severity_range={"min": 0.3, "max": 1.0}
            ),
            ProductRecommendation(
                name="Vitamin C Serum",
                brand="Skinceuticals",
                category="serum",
                description="Antioxidant serum for brightening and protection",
                key_ingredients=["l-ascorbic acid", "vitamin e", "ferulic acid"],
                skin_types=["normal", "dry", "combination"],
                concerns_addressed=["hyperpigmentation", "fine_lines"],
                severity_range={"min": 0.2, "max": 0.8}
            ),
            ProductRecommendation(
                name="Retinol Cream",
                brand="Neutrogena",
                category="treatment",
                description="Anti-aging retinol treatment for fine lines and texture",
                key_ingredients=["retinol", "hyaluronic acid"],
                skin_types=["normal", "combination", "oily"],
                concerns_addressed=["fine_lines", "acne"],
                contraindications=["sensitive"],
                severity_range={"min": 0.4, "max": 1.0}
            ),
            ProductRecommendation(
                name="Moisturizing Cream",
                brand="Vanicream",
                category="moisturizer",
                description="Gentle moisturizer for sensitive and dry skin",
                key_ingredients=["ceramides", "petrolatum", "dimethicone"],
                skin_types=["dry", "sensitive", "normal"],
                concerns_addressed=["dehydration", "redness"],
                severity_range={"min": 0.0, "max": 1.0}
            ),
            ProductRecommendation(
                name="Niacinamide Serum",
                brand="The Ordinary",
                category="serum",
                description="Reduces pore appearance and regulates sebum",
                key_ingredients=["niacinamide", "zinc"],
                skin_types=["oily", "combination", "normal"],
                concerns_addressed=["pore_size", "redness"],
                severity_range={"min": 0.3, "max": 0.9}
            ),
        ]
        
        for product in default_products:
            db.add(product)
        
        db.commit()
    
    db.close()

def get_database_status():
    """Check database connection status."""
    try:
        db = SessionLocal()
        # Simple query to test connection
        db.execute("SELECT 1")
        db.close()
        return {"status": "healthy", "message": "Database connection successful"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Database connection failed: {str(e)}"}

# Database health check
def database_health_check():
    """Perform comprehensive database health check."""
    try:
        db = SessionLocal()
        
        # Test basic connectivity
        result = db.execute("SELECT 1").fetchone()
        if not result:
            raise Exception("Basic query failed")
        
        # Check if tables exist
        from models import User, SkinAnalysis, ProductRecommendation
        user_count = db.query(User).count()
        analysis_count = db.query(SkinAnalysis).count()
        product_count = db.query(ProductRecommendation).count()
        
        db.close()
        
        return {
            "status": "healthy",
            "tables_exist": True,
            "user_count": user_count,
            "analysis_count": analysis_count,
            "product_count": product_count
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "tables_exist": False
        }