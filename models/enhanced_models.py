# models/enhanced_models.py
from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ProviderStatus(str, Enum):
    DISCOVERED = "discovered"
    ANALYZING = "analyzing" 
    VALIDATED = "validated"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"

class EnhancedProvider(SQLModel, table=True):
    """Enhanced provider model with AI capabilities"""
    
    id: Optional[int] = Field(primary_key=True)
    
    # Basic Information
    name: str = Field(index=True)
    website_url: str = Field(unique=True)
    country: str = Field(index=True)
    
    # AI Discovery Metadata
    discovery_method: str  # "ai_search", "registry_lookup", "competitor_analysis"
    discovery_confidence: float = Field(ge=0.0, le=1.0)
    discovery_timestamp: datetime
    
    # Provider Classification
    market_position: str  # "major", "regional", "niche", "startup"
    target_segment: str  # "B2C", "B2B", "both"
    company_size: Optional[str] = None  # "enterprise", "medium", "small"
    
    # Operational Status
    status: ProviderStatus = ProviderStatus.DISCOVERED
    last_analyzed: Optional[datetime] = None
    analysis_frequency_hours: int = Field(default=24)
    
    # Quality Metrics
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    reliability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Business Information (AI-extracted)
    business_registration_number: Optional[str] = None
    established_year: Optional[int] = None
    employee_count_estimate: Optional[str] = None
    annual_revenue_estimate: Optional[str] = None
    
    # Contact Information
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    customer_service_hours: Optional[str] = None
    
    # Relationships
    service_plans: List["EnhancedServicePlan"] = Relationship(back_populates="provider")
    reviews: List["ProviderReview"] = Relationship(back_populates="provider")

class EnhancedServicePlan(SQLModel, table=True):
    """Enhanced service plan with comprehensive details"""
    
    id: Optional[int] = Field(primary_key=True)
    
    # Provider Relationship
    provider_id: int = Field(foreign_key="enhancedprovider.id")
    provider: EnhancedProvider = Relationship(back_populates="service_plans")
    
    # Basic Plan Information
    name: str = Field(index=True)
    category: str = Field(index=True)
    subcategory: Optional[str] = None
    description: Optional[str] = None
    
    # Pricing Information
    monthly_price: Optional[float] = Field(ge=0.0)
    setup_fee: Optional[float] = Field(default=0.0, ge=0.0)
    cancellation_fee: Optional[float] = Field(default=0.0, ge=0.0)
    price_currency: str = Field(default="NOK")
    
    # Contract Terms
    contract_duration_months: Optional[int] = Field(ge=0)
    notice_period_days: Optional[int] = Field(ge=0)
    auto_renewal: Optional[bool] = None
    
    # Service-Specific Details (JSON storage for flexibility)
    technical_specifications: Optional[Dict[str, Any]] = Field(default={})
    included_features: Optional[List[str]] = Field(default=[])
    optional_addons: Optional[List[Dict[str, Any]]] = Field(default=[])
    
    # Market Analysis
    market_position_rank: Optional[int] = None
    price_competitiveness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    feature_richness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # AI Extraction Metadata
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    last_updated: datetime
    source_url: str
    extraction_method: str  # "ai_llm", "pattern", "api", "manual"
    
    # Compliance and Regulatory
    regulatory_compliance_status: Optional[str] = None
    certifications: Optional[List[str]] = Field(default=[])
    
    # Availability and Targeting
    geographic_availability: Optional[List[str]] = Field(default=[])
    target_customer_segments: Optional[List[str]] = Field(default=[])
    eligibility_requirements: Optional[List[str]] = Field(default=[])

class ProviderReview(SQLModel, table=True):
    """Comprehensive review aggregation"""
    
    id: Optional[int] = Field(primary_key=True)
    
    # Provider Relationship
    provider_id: int = Field(foreign_key="enhancedprovider.id")
    provider: EnhancedProvider = Relationship(back_populates="reviews")
    
    # Review Source Information
    source_platform: str  # "trustpilot", "google", "facebook", "reddit"
    source_url: str
    review_id_on_platform: Optional[str] = None
    
    # Review Content
    reviewer_name: Optional[str] = None
    review_title: Optional[str] = None
    review_text: Optional[str] = None
    rating: Optional[float] = Field(ge=0.0, le=5.0)
    review_date: Optional[datetime] = None
    
    # AI Analysis Results
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: str  # "positive", "negative", "neutral"
    authenticity_score: float = Field(ge=0.0, le=1.0)
    key_themes: List[str] = Field(default=[])
    mentioned_services: List[str] = Field(default=[])
    
    # Meta Information
    extraction_timestamp: datetime
    language_detected: str
    is_verified_purchase: Optional[bool] = None