# enhanced_main.py - Integration with existing architecture
import asyncio
from typing import Dict, List, Any
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import existing components
from services.ai_async_client import AIAsyncClient, create_ai_client
from config.settings import settings
from database import get_session, create_db_and_tables
from models import StandardizedProduct, ProductDB

# Import new AI agents
from agents.discovery_orchestrator import AdvancedDiscoveryOrchestrator, DiscoveryTarget
from agents.content_extraction_agent import IntelligentContentExtractor
from agents.market_intelligence_agent import MarketIntelligenceAgent

# Enhanced application with AI agents
class EnhancedAIAggregator:
    """Enhanced aggregator with multi-agent AI capabilities"""
    
    def __init__(self):
        self.ai_client: AIAsyncClient = None
        self.discovery_agent: AdvancedDiscoveryOrchestrator = None
        self.extraction_agent: IntelligentContentExtractor = None
        self.intelligence_agent: MarketIntelligenceAgent = None
        
    async def initialize(self):
        """Initialize all AI agents"""
        
        # Load AI configuration (reuse existing config)
        ai_config = {
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "tinyllama",
                "priority": 90,
                "is_free": True
            },
            "huggingface": {
                "api_key": settings.ai.get("huggingface_api_key"),
                "priority": 80,
                "is_free": True
            },
            "openai": {
                "api_key": settings.ai.get("openai_api_key"),
                "priority": 70,
                "cost_per_1k_tokens": 0.002
            }
        }
        
        # Initialize AI client (reuse existing implementation)
        self.ai_client = await create_ai_client(ai_config)
        
        # Initialize specialized agents
        self.discovery_agent = AdvancedDiscoveryOrchestrator(self.ai_client)
        self.extraction_agent = IntelligentContentExtractor(self.ai_client)
        self.intelligence_agent = MarketIntelligenceAgent(self.ai_client)
        
        print("✅ Enhanced AI Aggregator initialized with multi-agent system")
    
    async def autonomous_market_discovery(self, country: str, service_category: str) -> Dict[str, Any]:
        """
        Perform autonomous market discovery using AI agents
        """
        
        # Create discovery target
        target = DiscoveryTarget(
            country=country,
            service_category=service_category,
            language=self._get_country_language(country),
            currency=self._get_country_currency(country),
            regulatory_bodies=self._get_regulatory_bodies(country, service_category),
            market_size_estimate="medium",
            discovery_depth="comprehensive"
        )
        
        # Phase 1: Market intelligence gathering
        market_intel = await self.intelligence_agent.generate_market_intelligence(
            country, service_category
        )
        
        # Phase 2: Provider discovery
        async with self.discovery_agent as discovery:
            discovered_providers = await discovery.discover_market_comprehensive(target)
        
        # Phase 3: Content extraction for top candidates
        extraction_tasks = []
        for provider in discovered_providers[:10]:  # Top 10 candidates
            if provider.confidence_score > 0.8:
                task = self._extract_provider_data(provider)
                extraction_tasks.append(task)
        
        extracted_data = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        # Phase 4: Data standardization and storage
        standardized_products = []
        for data in extracted_data:
            if isinstance(data, Exception):
                continue
            
            # Convert to StandardizedProduct format
            products = self._convert_to_standardized_products(data, service_category)
            standardized_products.extend(products)
        
        return {
            "market_intelligence": market_intel,
            "discovered_providers": len(discovered_providers),
            "extracted_products": len(standardized_products),
            "products": standardized_products,
            "discovery_timestamp": datetime.now().isoformat()
        }

# Enhanced FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    # Startup
    await create_db_and_tables()
    
    # Initialize enhanced aggregator
    app.state.aggregator = EnhancedAIAggregator()
    await app.state.aggregator.initialize()
    
    yield
    
    # Shutdown
    if hasattr(app.state.aggregator, 'ai_client'):
        await app.state.aggregator.ai_client.close()

# Create FastAPI app with enhanced capabilities
app = FastAPI(
    title="Enhanced AI Service Aggregator",
    description="Next-generation AI-powered service discovery and aggregation platform",
    version="2.0.0",
    lifespan=lifespan
)

# Enhanced API endpoints
@app.post("/api/v2/autonomous-discovery")
async def autonomous_discovery(
    country: str,
    service_category: str,
    background_tasks: BackgroundTasks,
    session = Depends(get_session)
):
    """
    Perform autonomous market discovery using AI agents
    """
    
    try:
        # Start autonomous discovery
        result = await app.state.aggregator.autonomous_market_discovery(
            country=country,
            service_category=service_category
        )
        
        # Store discovered products in background
        if result.get("products"):
            background_tasks.add_task(
                store_discovered_products,
                session,
                result["products"]
            )
        
        return {
            "status": "success",
            "discovered_providers": result["discovered_providers"],
            "extracted_products": result["extracted_products"],
            "market_intelligence": result["market_intelligence"],
            "discovery_id": f"{country}_{service_category}_{int(time.time())}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")

@app.get("/api/v2/market-intelligence/{country}/{service_category}")
async def get_market_intelligence(country: str, service_category: str):
    """
    Get comprehensive market intelligence for specific market
    """
    
    try:
        market_intel = await app.state.aggregator.intelligence_agent.generate_market_intelligence(
            country=country,
            service_category=service_category
        )
        
        return {
            "status": "success",
            "market_intelligence": market_intel,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intelligence generation failed: {str(e)}")

# Run the enhanced application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "enhanced_main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers
    )