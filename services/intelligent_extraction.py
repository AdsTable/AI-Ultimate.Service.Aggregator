# services/intelligent_extraction.py
from crawl4ai import AsyncWebCrawler
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class IntelligentContentAnalyzer:
    """Advanced AI content analysis with deep understanding"""
    
    def __init__(self):
        self.crawler = AsyncWebCrawler()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = DeepSeek(api_key=os.getenv("DEEPSEEK_API_KEY"))
        
    async def extract_comprehensive_service_data(self, provider_url: str) -> Dict[str, Any]:
        """
        Extract comprehensive service data using multi-stage AI analysis
        """
        
        # Stage 1: Website structure analysis
        structure_analysis = await self._analyze_website_structure(provider_url)
        
        # Stage 2: Content extraction with context understanding
        content_data = await self._extract_contextual_content(
            provider_url, structure_analysis
        )
        
        # Stage 3: Cross-validation with external sources
        validated_data = await self._cross_validate_with_sources(
            content_data, provider_url
        )
        
        # Stage 4: Competitive analysis
        market_position = await self._analyze_market_position(validated_data)
        
        return {
            **validated_data,
            "market_analysis": market_position,
            "extraction_confidence": self._calculate_confidence_score(validated_data),
            "last_updated": datetime.now().isoformat()
        }
    
    async def _analyze_website_structure(self, url: str) -> Dict[str, Any]:
        """AI-powered website structure analysis"""
        
        result = await self.crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                extraction_strategy=LLMExtractionStrategy(
                    llm_config=LLMConfig(provider="deepseek/deepseek-chat"),
                    instruction="""
                    Analyze this website's structure and identify:
                    1. Service/product sections and their locations
                    2. Pricing information areas
                    3. Contact and company information
                    4. Terms and conditions sections
                    5. Customer review/testimonial areas
                    6. Navigation patterns and site architecture
                    
                    Focus on Norwegian service provider patterns.
                    Return structured analysis with CSS selectors and content types.
                    """,
                    schema={
                        "type": "object",
                        "properties": {
                            "services_section": {"type": "string"},
                            "pricing_section": {"type": "string"},
                            "contact_section": {"type": "string"},
                            "reviews_section": {"type": "string"},
                            "navigation_type": {"type": "string"},
                            "site_complexity": {"type": "string"},
                            "requires_js": {"type": "boolean"}
                        }
                    }
                )
            )
        )
        
        return json.loads(result.extracted_content) if result.extracted_content else {}