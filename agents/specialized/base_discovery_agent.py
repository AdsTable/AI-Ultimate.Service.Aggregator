# agents/specialized/base_discovery_agent.py
"""
Base class for all specialized discovery agents with shared functionality
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import json
from urllib.parse import urljoin, urlparse

from services.ai_async_client import AIAsyncClient

logger = logging.getLogger(__name__)

@dataclass
class DiscoveryResult:
    """Standardized discovery result across all agents"""
    provider_name: str
    website_url: str
    confidence_score: float  # 0.0 - 1.0
    discovery_method: str
    source_platform: str
    contact_info: Dict[str, Any]
    business_details: Dict[str, Any]
    market_indicators: Dict[str, Any]
    raw_data: Dict[str, Any]
    discovered_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'provider_name': self.provider_name,
            'website_url': self.website_url,
            'confidence_score': self.confidence_score,
            'discovery_method': self.discovery_method,
            'source_platform': self.source_platform,
            'contact_info': self.contact_info,
            'business_details': self.business_details,
            'market_indicators': self.market_indicators,
            'raw_data': self.raw_data,
            'discovered_at': self.discovered_at.isoformat()
        }

class BaseDiscoveryAgent(ABC):
    """
    Base class for all discovery agents with common functionality
    Implements shared patterns for caching, rate limiting, and error handling
    """
    
    def __init__(self, ai_client: AIAsyncClient, config: Dict[str, Any]):
        self.ai_client = ai_client
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.discovery_cache: Dict[str, List[DiscoveryResult]] = {}
        self.last_request_times: Dict[str, datetime] = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize HTTP session and agent-specific resources"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
        await self._initialize_agent_specific()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
        await self._cleanup_agent_specific()
    
    @abstractmethod
    async def _initialize_agent_specific(self):
        """Agent-specific initialization"""
        pass
    
    @abstractmethod
    async def _cleanup_agent_specific(self):
        """Agent-specific cleanup"""
        pass
    
    @abstractmethod
    async def discover_providers(self, target_country: str, service_category: str) -> List[DiscoveryResult]:
        """Main discovery method - must be implemented by each agent"""
        pass
    
    async def _apply_rate_limiting(self, platform: str, min_interval: float = 2.0):
        """Apply rate limiting for external API calls"""
        last_request = self.last_request_times.get(platform)
        if last_request:
            elapsed = (datetime.now() - last_request).total_seconds()
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s for {platform}")
                await asyncio.sleep(sleep_time)
        
        self.last_request_times[platform] = datetime.now()
    
    async def _get_cached_results(self, cache_key: str, max_age_hours: int = 24) -> Optional[List[DiscoveryResult]]:
        """Get cached discovery results if they're fresh enough"""
        if cache_key in self.discovery_cache:
            results = self.discovery_cache[cache_key]
            if results and (datetime.now() - results[0].discovered_at).total_seconds() < max_age_hours * 3600:
                logger.debug(f"Using cached results for {cache_key}")
                return results
        return None
    
    async def _cache_results(self, cache_key: str, results: List[DiscoveryResult]):
        """Cache discovery results"""
        self.discovery_cache[cache_key] = results
        logger.debug(f"Cached {len(results)} results for {cache_key}")
    
    async def _ai_analyze_text(self, text: str, analysis_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Common AI text analysis functionality"""
        try:
            prompt = f"""
            Analyze this {analysis_type} content for service provider information:
            
            Content: {text[:5000]}  # Limit to avoid token overflow
            Context: {json.dumps(context)}
            
            Extract and return JSON with:
            {{
                "is_service_provider": boolean,
                "provider_name": "extracted_name_or_null",
                "confidence_score": 0.0-1.0,
                "service_indicators": ["indicator1", "indicator2"],
                "contact_hints": {{"phone": "", "email": "", "website": ""}},
                "business_type": "B2C/B2B/both",
                "market_position": "major/regional/local/unknown",
                "reasoning": "brief_explanation"
            }}
            """
            
            response = await self.ai_client.ask(
                prompt=prompt,
                provider="ollama",  # Start with free provider
                task_complexity="medium"
            )
            
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {
                "is_service_provider": False,
                "confidence_score": 0.0,
                "reasoning": f"Analysis failed: {str(e)}"
            }
    
    def _create_discovery_result(self, 
                               provider_name: str,
                               website_url: str,
                               confidence_score: float,
                               discovery_method: str,
                               source_platform: str,
                               additional_data: Dict[str, Any]) -> DiscoveryResult:
        """Helper to create standardized discovery results"""
        return DiscoveryResult(
            provider_name=provider_name,
            website_url=website_url,
            confidence_score=confidence_score,
            discovery_method=discovery_method,
            source_platform=source_platform,
            contact_info=additional_data.get('contact_info', {}),
            business_details=additional_data.get('business_details', {}),
            market_indicators=additional_data.get('market_indicators', {}),
            raw_data=additional_data.get('raw_data', {}),
            discovered_at=datetime.now()
        )