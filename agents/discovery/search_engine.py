# agents/discovery/search_engine.py
"""
Search Engine Discovery Agent

This agent specializes in discovering service providers through search engines
using AI-powered query generation and result validation. It integrates with
various search APIs and uses intelligent result filtering.
"""

import asyncio
import logging
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin, quote_plus
import aiohttp
from bs4 import BeautifulSoup
import hashlib

from ..base import BaseAgent, AgentConfig, AgentError, safe_json_parse, retry_with_backoff, rate_limiter
from .models import DiscoveryTarget, ProviderCandidate, SearchStrategy
from services.ai_async_client import AIAsyncClient, TaskComplexity

logger = logging.getLogger(__name__)


class SearchEngineAgent(BaseAgent):
    """
    Specialized agent for search engine-based provider discovery
    
    Features:
    - Multi-search engine support (Google, Bing, DuckDuckGo)
    - AI-powered result validation and filtering
    - Website content analysis for provider verification
    - Intelligent query generation and optimization
    - Rate limiting and respectful crawling
    - Result caching and deduplication
    """
    
    def __init__(self, ai_client: AIAsyncClient, redis_client=None):
        config = AgentConfig(
            name="SearchEngineAgent",
            max_retries=2,
            rate_limit=20,  # Conservative rate limiting for search APIs
            preferred_ai_provider="ollama",  # Start with free provider
            task_complexity=TaskComplexity.MEDIUM,
            cache_ttl=1800,  # 30 minutes cache for search results
            debug=False
        )
        
        super().__init__(config, ai_client)
        
        # Redis client for caching search results
        self.redis_client = redis_client
        
        # Search engine configurations
        self.search_engines = {
            'duckduckgo': {
                'base_url': 'https://api.duckduckgo.com/',
                'rate_limit': 1.0,  # 1 second between requests
                'requires_api_key': False,
                'max_results': 10
            },
            'bing': {
                'base_url': 'https://api.bing.microsoft.com/v7.0/search',
                'rate_limit': 0.5,  # 0.5 seconds between requests
                'requires_api_key': True,
                'max_results': 20
            },
            'google': {
                'base_url': 'https://www.googleapis.com/customsearch/v1',
                'rate_limit': 1.0,  # 1 second between requests
                'requires_api_key': True,
                'max_results': 10
            }
        }
        
        # HTTP session for web requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Search result cache
        self.search_cache = {}
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'cached_results': 0,
            'validated_providers': 0,
            'search_engine_performance': {}
        }
    
    async def _setup_agent(self) -> None:
        """Initialize HTTP session and search engine configurations"""
        try:
            # Setup HTTP session with optimized settings
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'AIAggregator/1.0 (Research Bot; +https://github.com/AdsTable/AIagreggatorSevice)'
                }
            )
            
            # Load search results cache
            await self._load_search_cache()
            
            self.logger.info("Search engine agent initialized successfully")
            
        except Exception as e:
            raise AgentError(self.config.name, f"Failed to initialize search engine agent: {e}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup HTTP session and save cache"""
        try:
            # Save search cache
            await self._save_search_cache()
            
            # Close HTTP session
            if self.session and not self.session.closed:
                await self.session.close()
                await asyncio.sleep(0.1)  # Allow cleanup
            
            self.logger.info("Search engine agent cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup warning for search engine agent: {e}")
    
    async def discover_providers(
        self, 
        target: DiscoveryTarget, 
        strategies: List[SearchStrategy]
    ) -> List[ProviderCandidate]:
        """
        Main discovery method using search engines with AI validation
        
        Args:
            target: Discovery target specification
            strategies: Search strategies to execute
            
        Returns:
            List of validated provider candidates
        """
        self.logger.info(f"🔍 Starting search engine discovery for {target.service_category} in {target.country}")
        
        start_time = time.time()
        all_candidates = []
        
        try:
            # Filter for search engine strategies
            search_strategies = [s for s in strategies if s.method == 'search_engines']
            
            if not search_strategies:
                self.logger.warning("No search engine strategies provided")
                return []
            
            # Execute search strategies
            for strategy in search_strategies:
                strategy_candidates = await self._execute_search_strategy(target, strategy)
                all_candidates.extend(strategy_candidates)
                
                # Rate limiting between strategies
                await asyncio.sleep(1.0)
            
            # Deduplicate by website URL
            unique_candidates = await self._deduplicate_candidates(all_candidates)
            
            # Validate candidates using AI
            validated_candidates = await self._validate_candidates_with_ai(unique_candidates, target)
            
            discovery_time = time.time() - start_time
            self.search_stats['total_searches'] += 1
            if validated_candidates:
                self.search_stats['successful_searches'] += 1
            
            self.logger.info(
                f"✅ Search engine discovery complete: {len(validated_candidates)} validated candidates "
                f"found in {discovery_time:.1f}s"
            )
            
            return validated_candidates
            
        except Exception as e:
            self.logger.error(f"❌ Search engine discovery failed: {e}")
            raise AgentError(self.config.name, f"Search discovery failed: {e}")
    
    async def _execute_search_strategy(
        self, 
        target: DiscoveryTarget, 
        strategy: SearchStrategy
    ) -> List[ProviderCandidate]:
        """
        Execute a specific search strategy across multiple search engines
        
        Args:
            target: Discovery target
            strategy: Search strategy to execute
            
        Returns:
            List of candidate providers from search results
        """
        self.logger.info(f"🎯 Executing search strategy: {strategy.method} (priority: {strategy.priority})")
        
        candidates = []
        queries = strategy.queries
        
        # Optimize queries for the target market
        optimized_queries = await self._optimize_queries_for_target(queries, target)
        
        # Execute searches across available search engines
        for query in optimized_queries[:5]:  # Limit to top 5 queries to control costs
            try:
                # Try multiple search engines for redundancy
                search_engines_to_use = ['duckduckgo', 'bing']  # Start with free/cheap options
                
                for search_engine in search_engines_to_use:
                    engine_results = await self._execute_search_query(
                        query, search_engine, target.country, target.language
                    )
                    
                    if engine_results:
                        # Convert search results to candidates
                        query_candidates = await self._process_search_results(
                            engine_results, query, search_engine, target
                        )
                        candidates.extend(query_candidates)
                        
                        # If we got good results, don't need to try other engines for this query
                        if len(query_candidates) >= 3:
                            break
                    
                    # Rate limiting between search engines
                    await asyncio.sleep(self.search_engines[search_engine]['rate_limit'])
                
            except Exception as e:
                self.logger.error(f"Search query failed for '{query}': {e}")
                continue
        
        self.logger.info(f"📊 Strategy executed: {len(candidates)} raw candidates found")
        return candidates
    
    async def _optimize_queries_for_target(
        self, 
        base_queries: List[str], 
        target: DiscoveryTarget
    ) -> List[str]:
        """
        Use AI to optimize search queries for better results in the target market
        
        Args:
            base_queries: Original search queries
            target: Discovery target with market context
            
        Returns:
            Optimized search queries
        """
        
        optimization_prompt = f"""
        Optimize these search queries to find {target.service_category} providers in {target.country}.
        
        Original Queries:
        {json.dumps(base_queries, indent=2)}
        
        Target Market Context:
        - Country: {target.country}
        - Service Category: {target.service_category}
        - Language: {target.language}
        - Market Size: {target.market_size_estimate}
        
        Search Query Optimization Guidelines:
        1. **Geographic Specificity**: Add location modifiers for {target.country}
        2. **Language Adaptation**: Use {target.language} terms where appropriate
        3. **Industry Terminology**: Include sector-specific keywords
        4. **Search Operators**: Add effective Google/Bing search operators
        5. **Local Context**: Consider local business naming conventions
        
        Generate optimized queries:
        {{
          "optimized_queries": [
            "enhanced_query_1",
            "enhanced_query_2",
            "enhanced_query_3"
          ],
          "search_operators_used": ["site:", "intitle:", "filetype:"],
          "local_adaptations": ["geographic_terms", "language_adaptations"],
          "optimization_reasoning": "explanation_of_improvements"
        }}
        
        Return 6-8 highly targeted queries for maximum discovery effectiveness.
        """
        
        try:
            response = await self.ask_ai(
                prompt=optimization_prompt,
                provider="ollama",  # Use free provider for query optimization
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=800
            )
            
            optimization_data = safe_json_parse(response, default={"optimized_queries": base_queries})
            optimized_queries = optimization_data.get('optimized_queries', base_queries)
            
            # Combine original and optimized queries, removing duplicates
            all_queries = list(set(base_queries + optimized_queries))
            
            self.logger.info(f"🎯 Query optimization: {len(base_queries)} → {len(all_queries)} queries")
            return all_queries
            
        except Exception as e:
            self.logger.warning(f"Query optimization failed: {e}")
            return base_queries
    
    @rate_limiter(max_calls=30, time_window=60)  # 30 calls per minute
    async def _execute_search_query(
        self, 
        query: str, 
        search_engine: str,
        country: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Execute a search query on a specific search engine
        
        Args:
            query: Search query string
            search_engine: Name of search engine to use
            country: Target country for geo-specific results
            language: Target language for results
            
        Returns:
            List of search result dictionaries
        """
        
        # Check cache first
        cache_key = hashlib.md5(f"{search_engine}_{query}_{country}".encode()).hexdigest()
        cached_results = await self._get_cached_search_results(cache_key)
        if cached_results:
            self.search_stats['cached_results'] += 1
            self.logger.debug(f"📚 Using cached results for query: {query[:50]}...")
            return cached_results
        
        try:
            if search_engine == 'duckduckgo':
                results = await self._search_duckduckgo(query, country, language)
            elif search_engine == 'bing':
                results = await self._search_bing(query, country, language)
            elif search_engine == 'google':
                results = await self._search_google(query, country, language)
            else:
                self.logger.warning(f"Unknown search engine: {search_engine}")
                return []
            
            # Cache successful results
            if results:
                await self._cache_search_results(cache_key, results)
            
            # Update search engine performance stats
            if search_engine not in self.search_stats['search_engine_performance']:
                self.search_stats['search_engine_performance'][search_engine] = {
                    'total_queries': 0,
                    'successful_queries': 0,
                    'avg_results': 0.0
                }
            
            engine_stats = self.search_stats['search_engine_performance'][search_engine]
            engine_stats['total_queries'] += 1
            if results:
                engine_stats['successful_queries'] += 1
                prev_avg = engine_stats['avg_results']
                successful = engine_stats['successful_queries']
                engine_stats['avg_results'] = (prev_avg * (successful - 1) + len(results)) / successful
            
            self.logger.debug(f"🔍 {search_engine}: {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self.logger.error(f"Search query failed on {search_engine}: {e}")
            return []
    
    async def _search_duckduckgo(
        self, 
        query: str, 
        country: str, 
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Search using DuckDuckGo Instant Answer API (free)
        
        Note: DuckDuckGo's API is limited, so this implementation uses
        a simplified approach for demonstration purposes.
        """
        try:
            # DuckDuckGo Instant Answer API (limited but free)
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            url = 'https://api.duckduckgo.com/'
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract useful results from DuckDuckGo response
                    results = []
                    
                    # Check for related topics (companies)
                    related_topics = data.get('RelatedTopics', [])
                    for topic in related_topics[:10]:  # Limit to 10 results
                        if isinstance(topic, dict) and 'FirstURL' in topic:
                            result = {
                                'title': topic.get('Text', '').split(' - ')[0],
                                'url': topic.get('FirstURL', ''),
                                'snippet': topic.get('Text', ''),
                                'source': 'duckduckgo'
                            }
                            results.append(result)
                    
                    # Also check abstract if available
                    if data.get('Abstract') and data.get('AbstractURL'):
                        result = {
                            'title': data.get('Heading', ''),
                            'url': data.get('AbstractURL', ''),
                            'snippet': data.get('Abstract', ''),
                            'source': 'duckduckgo'
                        }
                        results.append(result)
                    
                    return results
                else:
                    self.logger.warning(f"DuckDuckGo API returned status {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    async def _search_bing(
        self, 
        query: str, 
        country: str, 
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Search using Bing Search API (requires API key)
        
        Note: This requires a Bing Search API key. In production,
        you would configure this through environment variables.
        """
        try:
            # This would require a real Bing API key in production
            # For demonstration, return simulated results
            self.logger.info(f"🔍 Simulating Bing search for: {query}")
            
            # Simulate realistic search results
            simulated_results = [
                {
                    'title': f"Leading {query.split()[0]} Provider",
                    'url': f"https://example-{query.split()[0].lower()}.com",
                    'snippet': f"Professional {query} services in {country}. Trusted by thousands of clients.",
                    'source': 'bing'
                },
                {
                    'title': f"Top {query.split()[0]} Companies in {country}",
                    'url': f"https://top-{query.split()[0].lower()}-{country.lower()}.com",
                    'snippet': f"Directory of verified {query} providers serving {country}.",
                    'source': 'bing'
                }
            ]
            
            return simulated_results
            
        except Exception as e:
            self.logger.error(f"Bing search failed: {e}")
            return []
    
    async def _search_google(
        self, 
        query: str, 
        country: str, 
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Search using Google Custom Search API (requires API key)
        
        Note: This requires Google Custom Search API credentials.
        For demonstration, returns simulated results.
        """
        try:
            # This would require real Google API credentials in production
            # For demonstration, return simulated results
            self.logger.info(f"🔍 Simulating Google search for: {query}")
            
            # Simulate realistic search results
            simulated_results = [
                {
                    'title': f"Professional {query.split()[0]} Services",
                    'url': f"https://pro-{query.split()[0].lower()}.{country.lower()}",
                    'snippet': f"Expert {query} solutions for businesses in {country}. Contact us today.",
                    'source': 'google'
                },
                {
                    'title': f"{query.split()[0]} Directory {country}",
                    'url': f"https://{query.split()[0].lower()}-directory.com/{country.lower()}",
                    'snippet': f"Complete listing of {query} providers in {country}.",
                    'source': 'google'
                }
            ]
            
            return simulated_results
            
        except Exception as e:
            self.logger.error(f"Google search failed: {e}")
            return []
    
    async def _process_search_results(
        self, 
        search_results: List[Dict[str, Any]], 
        query: str,
        search_engine: str,
        target: DiscoveryTarget
    ) -> List[ProviderCandidate]:
        """
        Process raw search results into provider candidates
        
        Args:
            search_results: Raw search results from search engine
            query: Original search query
            search_engine: Source search engine
            target: Discovery target
            
        Returns:
            List of provider candidates
        """
        candidates = []
        
        for i, result in enumerate(search_results):
            try:
                # Extract basic information
                title = result.get('title', '')
                url = result.get('url', '')
                snippet = result.get('snippet', '')
                
                # Skip invalid results
                if not url or not title:
                    continue
                
                # Parse domain for basic validation
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower()
                
                # Skip obviously irrelevant domains
                if any(skip in domain for skip in ['wikipedia.org', 'youtube.com', 'facebook.com', 'linkedin.com']):
                    continue
                
                # Create candidate with basic information
                candidate = ProviderCandidate(
                    name=self._extract_company_name_from_title(title),
                    website=url,
                    discovery_method=f'search_engine_{search_engine}',
                    confidence_score=0.5,  # Base confidence, will be improved by AI validation
                    business_category=target.service_category,
                    market_position='unknown',
                    ai_analysis={
                        'search_engine': search_engine,
                        'search_query': query,
                        'result_position': i + 1,
                        'title': title,
                        'snippet': snippet,
                        'domain': domain,
                        'needs_validation': True
                    }
                )
                
                candidates.append(candidate)
                
            except Exception as e:
                self.logger.warning(f"Failed to process search result: {e}")
                continue
        
        return candidates
    
    def _extract_company_name_from_title(self, title: str) -> str:
        """
        Extract likely company name from search result title
        
        Args:
            title: Search result title
            
        Returns:
            Extracted company name
        """
        if not title:
            return "Unknown"
        
        # Clean up common title patterns
        title = title.strip()
        
        # Remove common suffixes
        suffixes_to_remove = [
            ' - Home', ' | Home', ' - Official Site', ' | Official Site',
            ' - Services', ' | Services', ' - About Us', ' | About Us'
        ]
        
        for suffix in suffixes_to_remove:
            if title.endswith(suffix):
                title = title[:-len(suffix)]
                break
        
        # Extract first part if there's a separator
        separators = [' - ', ' | ', ' :: ', ' — ']
        for separator in separators:
            if separator in title:
                title = title.split(separator)[0]
                break
        
        # Limit length and clean
        title = title[:100].strip()
        
        return title if title else "Unknown"
    
    async def _deduplicate_candidates(
        self, 
        candidates: List[ProviderCandidate]
    ) -> List[ProviderCandidate]:
        """
        Remove duplicate candidates based on website URL
        
        Args:
            candidates: List of provider candidates
            
        Returns:
            Deduplicated list of candidates
        """
        seen_domains = set()
        unique_candidates = []
        
        for candidate in candidates:
            try:
                # Parse domain from website URL
                parsed_url = urlparse(candidate.website)
                domain = parsed_url.netloc.lower()
                
                # Remove 'www.' prefix for comparison
                if domain.startswith('www.'):
                    domain = domain[4:]
                
                if domain not in seen_domains and domain:
                    seen_domains.add(domain)
                    unique_candidates.append(candidate)
                    
            except Exception as e:
                self.logger.warning(f"Error processing candidate URL {candidate.website}: {e}")
                # Include candidate anyway if URL parsing fails
                unique_candidates.append(candidate)
        
        self.logger.info(f"🔄 Deduplication: {len(candidates)} → {len(unique_candidates)} unique candidates")
        return unique_candidates
    
    async def _validate_candidates_with_ai(
        self, 
        candidates: List[ProviderCandidate], 
        target: DiscoveryTarget
    ) -> List[ProviderCandidate]:
        """
        Use AI to validate and enhance candidate information
        
        Args:
            candidates: List of candidates to validate
            target: Discovery target for validation context
            
        Returns:
            List of validated and enhanced candidates
        """
        if not candidates:
            return []
        
        self.logger.info(f"🤖 Validating {len(candidates)} candidates with AI")
        
        validated_candidates = []
        
        # Process candidates in batches to manage token limits
        batch_size = 5
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            
            try:
                batch_validated = await self._validate_candidate_batch(batch, target)
                validated_candidates.extend(batch_validated)
                
                # Rate limiting between batches
                await asyncio.sleep(2.0)
                
            except Exception as e:
                self.logger.error(f"Batch validation failed: {e}")
                # Include unvalidated candidates as fallback
                for candidate in batch:
                    candidate.ai_analysis['validation_status'] = 'failed'
                    validated_candidates.append(candidate)
        
        # Filter out low-confidence candidates
        high_confidence_candidates = [
            c for c in validated_candidates 
            if c.confidence_score >= 0.6
        ]
        
        self.search_stats['validated_providers'] += len(high_confidence_candidates)
        
        self.logger.info(
            f"✅ Validation complete: {len(high_confidence_candidates)} high-confidence candidates "
            f"(threshold: 0.6)"
        )
        
        return high_confidence_candidates
    
    async def _validate_candidate_batch(
        self, 
        batch: List[ProviderCandidate], 
        target: DiscoveryTarget
    ) -> List[ProviderCandidate]:
        """
        Validate a batch of candidates using AI analysis
        
        Args:
            batch: Batch of candidates to validate
            target: Discovery target for context
            
        Returns:
            List of validated candidates
        """
        
        # Prepare batch data for AI analysis
        batch_data = []
        for i, candidate in enumerate(batch):
            batch_data.append({
                'id': i,
                'name': candidate.name,
                'website': candidate.website,
                'title': candidate.ai_analysis.get('title', ''),
                'snippet': candidate.ai_analysis.get('snippet', ''),
                'domain': candidate.ai_analysis.get('domain', ''),
                'search_engine': candidate.ai_analysis.get('search_engine', '')
            })
        
        validation_prompt = f"""
        Validate these potential {target.service_category} providers discovered through search engines.
        
        Target Market: {target.service_category} in {target.country}
        Language: {target.language}
        
        Candidates to Validate:
        {json.dumps(batch_data, indent=2)}
        
        For each candidate, analyze:
        1. **Business Legitimacy**: Does this appear to be a real business?
        2. **Service Relevance**: Do they offer {target.service_category} services?
        3. **Geographic Relevance**: Do they serve {target.country}?
        4. **Website Quality**: Professional website indicating legitimate business?
        5. **Search Context**: Does the search result context make sense?
        
        Validation Criteria:
        - High Confidence (0.8-1.0): Clearly legitimate provider with strong indicators
        - Medium Confidence (0.6-0.7): Likely legitimate with some positive indicators
        - Low Confidence (0.3-0.5): Uncertain legitimacy, mixed signals
        - Very Low Confidence (0.0-0.2): Likely not a legitimate provider
        
        Return JSON validation results:
        {{
          "validated_candidates": [{{
            "candidate_id": 0,
            "is_legitimate_provider": true/false,
            "confidence_score": 0.0-1.0,
            "enhanced_name": "improved_company_name",
            "market_position": "leader/regional/niche/startup/unknown",
            "legitimacy_indicators": ["professional_website", "clear_services"],
            "concerns": ["potential_issues"],
            "geographic_coverage": ["regions_served"],
            "service_validation": {{
              "offers_target_services": true/false,
              "service_focus": ["primary_services"],
              "specializations": ["niche_areas"]
            }},
            "validation_reasoning": "detailed_explanation"
          }}]
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=validation_prompt,
                provider="huggingface",  # Use free provider for validation
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1500
            )
            
            validation_data = safe_json_parse(response, default={"validated_candidates": []})
            validated_data = validation_data.get('validated_candidates', [])
            
            # Apply validation results to candidates
            validated_candidates = []
            for validation in validated_data:
                try:
                    candidate_id = validation.get('candidate_id')
                    if candidate_id is not None and candidate_id < len(batch):
                        candidate = batch[candidate_id]
                        
                        # Update candidate with validation results
                        candidate.confidence_score = validation.get('confidence_score', 0.3)
                        candidate.name = validation.get('enhanced_name', candidate.name)
                        candidate.market_position = validation.get('market_position', 'unknown')
                        
                        # Update AI analysis
                        candidate.ai_analysis.update({
                            'validation_status': 'completed',
                            'is_legitimate': validation.get('is_legitimate_provider', False),
                            'legitimacy_indicators': validation.get('legitimacy_indicators', []),
                            'concerns': validation.get('concerns', []),
                            'geographic_coverage': validation.get('geographic_coverage', []),
                            'service_validation': validation.get('service_validation', {}),
                            'validation_reasoning': validation.get('validation_reasoning', '')
                        })
                        
                        # Only include legitimate providers
                        if validation.get('is_legitimate_provider', False):
                            validated_candidates.append(candidate)
                            
                except Exception as e:
                    self.logger.warning(f"Failed to apply validation to candidate {candidate_id}: {e}")
            
            return validated_candidates
            
        except Exception as e:
            self.logger.error(f"AI validation failed: {e}")
            # Return original candidates with validation failure flag
            for candidate in batch:
                candidate.ai_analysis['validation_status'] = 'ai_failed'
                candidate.confidence_score = max(0.3, candidate.confidence_score)
            return batch
    
    async def _get_cached_search_results(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached search results"""
        if cache_key in self.search_cache:
            cached_data = self.search_cache[cache_key]
            # Check if cache is still valid (30 minutes)
            if time.time() - cached_data['timestamp'] < self.config.cache_ttl:
                return cached_data['results']
            else:
                del self.search_cache[cache_key]
        
        # Try Redis cache if available
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"search:{cache_key}")
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                self.logger.warning(f"Failed to get cached search results: {e}")
        
        return None
    
    async def _cache_search_results(self, cache_key: str, results: List[Dict[str, Any]]):
        """Cache search results for reuse"""
        # Cache in memory
        self.search_cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }
        
        # Cache in Redis if available
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"search:{cache_key}",
                    self.config.cache_ttl,
                    json.dumps(results)
                )
            except Exception as e:
                self.logger.warning(f"Failed to cache search results: {e}")
    
    async def _load_search_cache(self):
        """Load search cache from persistent storage"""
        if not self.redis_client:
            return
        
        try:
            cache_data = await self.redis_client.get("search_engine_cache")
            if cache_data:
                self.search_cache = json.loads(cache_data)
                self.logger.info("📚 Loaded search cache from storage")
        except Exception as e:
            self.logger.warning(f"Failed to load search cache: {e}")
    
    async def _save_search_cache(self):
        """Save search cache to persistent storage"""
        if not self.redis_client:
            return
        
        try:
            # Clean expired entries before saving
            current_time = time.time()
            cleaned_cache = {
                k: v for k, v in self.search_cache.items()
                if current_time - v['timestamp'] < self.config.cache_ttl
            }
            
            await self.redis_client.setex(
                "search_engine_cache",
                86400,  # 24 hours
                json.dumps(cleaned_cache)
            )
        except Exception as e:
            self.logger.warning(f"Failed to save search cache: {e}")
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get comprehensive search engine statistics"""
        base_stats = self.get_metrics()
        
        search_stats = {
            **base_stats,
            "search_performance": self.search_stats,
            "cache_size": len(self.search_cache),
            "available_engines": list(self.search_engines.keys()),
            "engine_configurations": {
                name: {
                    'rate_limit': config['rate_limit'],
                    'max_results': config['max_results'],
                    'requires_api_key': config['requires_api_key']
                }
                for name, config in self.search_engines.items()
            }
        }
        
        return search_stats