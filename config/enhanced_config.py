# config/enhanced_config.py
class EnhancedAIConfig:
    """Enhanced configuration for autonomous AI aggregator"""
    
    # AI Models Configuration
    PRIMARY_LLM = "deepseek/deepseek-chat"
    BACKUP_LLM = "ollama/llama2"  # Local fallback
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Discovery Settings
    AUTO_DISCOVERY_ENABLED = True
    DISCOVERY_SCHEDULE = "0 2 * * *"  # Daily at 2 AM
    MAX_CONCURRENT_DISCOVERIES = 10
    
    # Market Coverage
    SUPPORTED_COUNTRIES = ["Norway", "Sweden", "Denmark"]
    SERVICE_CATEGORIES = {
        "energy": {
            "subcategories": ["electricity", "gas", "renewable"],
            "regulatory_bodies": ["NVE", "RME"],
            "key_directories": ["kraftpriiser.no", "energimarknaden.se"]
        },
        "telecom": {
            "subcategories": ["mobile", "broadband", "fiber", "landline"],
            "regulatory_bodies": ["Nkom", "PTS"],
            "key_directories": ["mobilabonnement.no", "bredbandskollen.se"]
        },
        "financial": {
            "subcategories": ["insurance", "loans", "banking", "investment"],
            "regulatory_bodies": ["Finanstilsynet", "FI"],
            "key_directories": ["finansportalen.no", "konsumentverket.se"]
        }
    }
    
    # Quality Thresholds
    MIN_PROVIDER_CONFIDENCE = 0.75
    MIN_REVIEW_AUTHENTICITY = 0.60
    MIN_DATA_COMPLETENESS = 0.80