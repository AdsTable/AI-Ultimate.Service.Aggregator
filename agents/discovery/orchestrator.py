# agents/discovery/orchestrator.py
"""
Advanced Discovery Orchestrator - Main coordination agent for autonomous market discovery

This orchestrator coordinates multiple specialized discovery agents to find service
providers in specific markets using AI-powered analysis and validation. It implements
intelligent search strategy generation, parallel discovery execution, and self-learning
pattern improvement.

Features:
- Multi-source discovery using 6 parallel methods
- AI-powered search strategy generation
- Intelligent provider validation and confidence scoring
- Cost-optimized AI usage with provider routing
- Self-learning discovery pattern improvement
- Comprehensive error handling and fallback mechanisms
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict
import hashlib

from ..base import BaseAgent, AgentConfig, AgentError, safe_json_parse, retry_with_backoff
from .models import DiscoveryTarget, ProviderCandidate, SearchStrategy
from .search_engine_agent import SearchEngineAgent
from .regulatory_scanner import RegulatoryScanner
from .competitor_analyzer import CompetitorAnalyzer
from .social_intelligence_agent import SocialIntelligenceAgent
from services.ai_async_client import AIAsyncClient, TaskComplexity

logger = logging.getLogger(__name__)


class AdvancedDiscoveryOrchestrator(BaseAgent):
    """
    AI-powered discovery orchestrator with intelligent search strategies
    
    This orchestrator manages the entire discovery process, coordinating multiple
    specialized agents to comprehensively discover service providers in target markets.
    It uses AI to generate intelligent search strategies, validate results, and learn
    from successful discovery patterns.
    
    Cost Optimization:
    - Uses Ollama for strategy generation (free)
    - Leverages HuggingFace for content analysis (free)
    - Falls back to Groq/OpenAI only for complex analysis
    - Implements caching to reduce API calls
    """
    
    def __init__(self, ai_client: AIAsyncClient, redis_client=None):
        config = AgentConfig(
            name="DiscoveryOrchestrator",
            max_retries=3,
            rate_limit=30,  # Conservative rate limiting for stability
            preferred_ai_provider="ollama",  # Start with free provider
            task_complexity=TaskComplexity.COMPLEX,
            cache_ttl=3600,  # 1 hour cache for discovery strategies
            debug=False
        )
        
        super().__init__(config, ai_client)
        
        # Redis client for caching and learning patterns
        self.redis_client = redis_client
        
        # Initialize specialized discovery agents
        self.specialized_agents = {}
        
        # Discovery pattern learning storage
        self.discovery_patterns_cache = {}
        self.successful_strategies_cache = {}
        
        # Performance tracking
        self.discovery_stats = {
            'total_discoveries': 0,
            'successful_discoveries': 0,
            'avg_discovery_time': 0.0,
            'avg_candidates_found': 0.0,
            'method_performance': {}
        }
    
    async def _setup_agent(self) -> None:
        """Initialize specialized discovery agents and setup resources"""
        try:
            # Initialize specialized agents with shared AI client
            self.specialized_agents = {
                'search_engine': SearchEngineAgent(self.ai_client, self.redis_client),
                'regulatory': RegulatoryScanner(self.ai_client, self.redis_client),
                'competitor': CompetitorAnalyzer(self.ai_client, self.redis_client),
                'social_intel': SocialIntelligenceAgent(self.ai_client, self.redis_client)
            }
            
            # Initialize all specialized agents
            initialization_tasks = []
            for agent_name, agent in self.specialized_agents.items():
                task = self._initialize_agent_safely(agent_name, agent)
                initialization_tasks.append(task)
            
            # Wait for all agents to initialize
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # Check initialization results
            failed_agents = []
            for i, (agent_name, result) in enumerate(zip(self.specialized_agents.keys(), results)):
                if isinstance(result, Exception):
                    failed_agents.append(agent_name)
                    self.logger.error(f"Failed to initialize {agent_name} agent: {result}")
            
            if failed_agents:
                self.logger.warning(f"Some agents failed to initialize: {failed_agents}")
                # Remove failed agents from active pool
                for agent_name in failed_agents:
                    if agent_name in self.specialized_agents:
                        del self.specialized_agents[agent_name]
            
            # Load discovery patterns from cache
            await self._load_discovery_patterns()
            
            self.logger.info(f"Discovery orchestrator initialized with {len(self.specialized_agents)} active agents")
            
        except Exception as e:
            raise AgentError(self.config.name, f"Failed to initialize discovery orchestrator: {e}")
    
    async def _initialize_agent_safely(self, agent_name: str, agent) -> None:
        """Safely initialize an agent with timeout and error handling"""
        try:
            # Set timeout for agent initialization
            await asyncio.wait_for(agent.initialize(), timeout=30.0)
            self.logger.info(f"✅ {agent_name} agent initialized successfully")
        except asyncio.TimeoutError:
            raise AgentError(agent_name, f"{agent_name} agent initialization timed out")
        except Exception as e:
            raise AgentError(agent_name, f"{agent_name} agent initialization failed: {e}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup specialized agents and save learning patterns"""
        try:
            # Save discovery patterns to cache
            await self._save_discovery_patterns()
            
            # Cleanup specialized agents
            cleanup_tasks = []
            for agent_name, agent in self.specialized_agents.items():
                task = self._cleanup_agent_safely(agent_name, agent)
                cleanup_tasks.append(task)
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            self.logger.info("Discovery orchestrator cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup warning for discovery orchestrator: {e}")
    
    async def _cleanup_agent_safely(self, agent_name: str, agent) -> None:
        """Safely cleanup an agent with timeout"""
        try:
            await asyncio.wait_for(agent.cleanup(), timeout=10.0)
        except Exception as e:
            self.logger.warning(f"Cleanup warning for {agent_name}: {e}")
    
    async def execute(
        self, 
        target: DiscoveryTarget,
        selected_methods: Optional[List[str]] = None,
        max_candidates: int = 50
    ) -> List[ProviderCandidate]:
        """
        Main execution method for comprehensive market discovery
        
        Args:
            target: Discovery target specification
            selected_methods: Optional list of discovery methods to use
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of validated provider candidates
        """
        return await self.discover_market_comprehensive(target, selected_methods, max_candidates)
    
    async def discover_market_comprehensive(
        self, 
        target: DiscoveryTarget,
        selected_methods: Optional[List[str]] = None,
        max_candidates: int = 50
    ) -> List[ProviderCandidate]:
        """
        Comprehensive market discovery using multiple AI-guided methods
        
        This is the main discovery workflow that orchestrates all specialized agents
        to find service providers in the target market.
        
        Args:
            target: Market discovery target with country, service category, etc.
            selected_methods: Specific discovery methods to use (default: all available)
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of validated and ranked provider candidates
        """
        discovery_start_time = time.time()
        discovery_id = self._generate_discovery_id(target)
        
        self.logger.info(
            f"🚀 Starting comprehensive discovery [{discovery_id}] for "
            f"{target.service_category} in {target.country} (depth: {target.discovery_depth})"
        )
        
        try:
            # Phase 1: Generate intelligent search strategies using AI
            self.logger.info(f"📋 Phase 1: Generating AI search strategies")
            search_strategies = await self._generate_ai_search_strategies(target)
            
            # Phase 2: Execute discovery methods in parallel
            self.logger.info(f"🔍 Phase 2: Executing parallel discovery methods")
            discovery_results = await self._execute_parallel_discovery(
                target, search_strategies, selected_methods
            )
            
            # Phase 3: Consolidate and validate results
            self.logger.info(f"🔄 Phase 3: Consolidating and validating results")
            all_candidates = await self._consolidate_discovery_results(discovery_results)
            
            # Phase 4: AI-powered deduplication and ranking
            self.logger.info(f"🎯 Phase 4: AI deduplication and ranking")
            final_candidates = await self._ai_deduplicate_and_rank(all_candidates, target, max_candidates)
            
            # Phase 5: Update learning patterns and statistics
            self.logger.info(f"📚 Phase 5: Updating learning patterns")
            discovery_time = time.time() - discovery_start_time
            await self._update_discovery_patterns(target, final_candidates, search_strategies, discovery_time)
            
            # Update performance statistics
            self._update_performance_stats(len(final_candidates), discovery_time)
            
            self.logger.info(
                f"✅ Discovery complete [{discovery_id}]: {len(final_candidates)} high-quality "
                f"candidates found in {discovery_time:.1f}s"
            )
            
            return final_candidates
            
        except Exception as e:
            discovery_time = time.time() - discovery_start_time
            self.logger.error(f"❌ Comprehensive discovery failed [{discovery_id}]: {e}")
            
            # Update failure statistics
            self.discovery_stats['total_discoveries'] += 1
            
            # Try to return partial results if available
            try:
                # Attempt basic fallback discovery
                fallback_candidates = await self._fallback_discovery(target, max_candidates)
                if fallback_candidates:
                    self.logger.info(f"🔄 Returning {len(fallback_candidates)} fallback candidates")
                    return fallback_candidates
            except Exception as fallback_error:
                self.logger.error(f"Fallback discovery also failed: {fallback_error}")
            
            raise AgentError(self.config.name, f"Discovery failed: {e}")
    
    def _generate_discovery_id(self, target: DiscoveryTarget) -> str:
        """Generate unique identifier for discovery session"""
        target_hash = hashlib.md5(
            f"{target.country}_{target.service_category}_{target.discovery_depth}".encode()
        ).hexdigest()[:8]
        timestamp = int(time.time())
        return f"DISC_{target_hash}_{timestamp}"
    
    async def _generate_ai_search_strategies(self, target: DiscoveryTarget) -> List[SearchStrategy]:
        """
        Generate intelligent search strategies using AI
        
        This method uses AI to analyze the target market and generate comprehensive
        search strategies tailored to the specific country, service category, and
        market characteristics.
        
        Cost Optimization: Uses Ollama (free) for strategy generation
        """
        
        # Check cache first
        cache_key = f"strategies_{target.country}_{target.service_category}_{target.discovery_depth}"
        cached_strategies = await self._get_cached_strategies(cache_key)
        if cached_strategies:
            self.logger.info("📚 Using cached search strategies")
            return cached_strategies
        
        strategy_prompt = f"""
        You are an expert market researcher specializing in {target.service_category} services in {target.country}.
        
        Generate comprehensive search strategies to find ALL service providers in this market.
        
        Target Market Analysis:
        - Country: {target.country}
        - Service Category: {target.service_category}
        - Language: {target.language}
        - Currency: {target.currency}
        - Regulatory Bodies: {', '.join(target.regulatory_bodies)}
        - Market Size Estimate: {target.market_size_estimate}
        - Discovery Depth: {target.discovery_depth}
        
        Consider these discovery vectors:
        1. **Search Engines**: Google, Bing, regional search engines specific to {target.country}
        2. **Official Channels**: Regulatory databases, business registries, licensing authorities
        3. **Industry Networks**: Trade associations, professional organizations, directories
        4. **Digital Footprint**: Social media, review platforms, job postings, news mentions
        5. **Competitive Intelligence**: Known competitors, market leaders, partnerships
        6. **Local Context**: Regional variations, local business customs, language nuances
        
        Generate specific strategies for each discovery method:
        
        {{
          "search_engines": {{
            "priority": 1-10,
            "queries": ["specific search terms in {target.language}"],
            "platforms": ["google.{target.country.lower()}", "bing.com", "local_engines"],
            "expected_yield": "estimated_number_of_providers",
            "search_operators": ["site: operators", "filetype: operators"],
            "geographic_modifiers": ["city names", "region names"]
          }},
          "regulatory_bodies": {{
            "priority": 1-10,
            "databases": ["specific database URLs or names"],
            "search_parameters": ["license types", "registration categories"],
            "data_sources": ["government portals", "professional boards"],
            "expected_yield": "estimated_number_of_providers"
          }},
          "competitor_analysis": {{
            "priority": 1-10,
            "seed_companies": ["known major players in {target.country}"],
            "analysis_methods": ["partnership analysis", "supplier networks"],
            "competitive_tools": ["seo analysis", "ad intelligence"],
            "expected_yield": "estimated_number_of_providers"
          }},
          "social_intelligence": {{
            "priority": 1-10,
            "platforms": ["linkedin", "facebook", "local_social_networks"],
            "search_terms": ["industry hashtags", "professional groups"],
            "content_types": ["company pages", "professional profiles"],
            "expected_yield": "estimated_number_of_providers"
          }},
          "industry_reports": {{
            "priority": 1-10,
            "report_sources": ["industry associations", "government reports"],
            "keywords": ["market analysis terms", "industry surveys"],
            "publication_types": ["white papers", "market studies"],
            "expected_yield": "estimated_number_of_providers"
          }},
          "news_analysis": {{
            "priority": 1-10,
            "news_sources": ["local business news", "industry publications"],
            "keywords": ["company announcements", "market developments"],
            "timeframe": ["recent 12 months", "industry milestones"],
            "expected_yield": "estimated_number_of_providers"
          }}
        }}
        
        Return valid JSON with detailed, actionable strategies for {target.country} market.
        """
        
        try:
            # Use free Ollama provider for strategy generation
            response = await self.ask_ai(
                prompt=strategy_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2048
            )
            
            strategies_data = safe_json_parse(response, default={})
            
            if not strategies_data:
                self.logger.warning("AI returned empty strategies, using fallback")
                return self._get_fallback_strategies(target)
            
            # Convert to SearchStrategy objects
            strategies = []
            for method_name, strategy_data in strategies_data.items():
                try:
                    strategy = SearchStrategy(
                        method=method_name,
                        priority=strategy_data.get('priority', 5),
                        queries=strategy_data.get('queries', []),
                        platforms=strategy_data.get('platforms', []),
                        expected_yield=strategy_data.get('expected_yield', 'unknown'),
                        ai_analysis_needed=True,
                        follow_up_actions=strategy_data.get('follow_up_actions', []),
                        metadata=strategy_data  # Store full strategy data
                    )
                    strategies.append(strategy)
                except Exception as e:
                    self.logger.warning(f"Failed to parse strategy for {method_name}: {e}")
            
            if not strategies:
                self.logger.warning("No valid strategies generated, using fallback")
                return self._get_fallback_strategies(target)
            
            # Enhance with memory from previous discoveries
            enhanced_strategies = await self._enhance_strategies_with_memory(strategies, target)
            
            # Cache successful strategies
            await self._cache_strategies(cache_key, enhanced_strategies)
            
            self.logger.info(f"✅ Generated {len(enhanced_strategies)} AI-powered discovery strategies")
            return enhanced_strategies
            
        except Exception as e:
            self.logger.error(f"❌ Strategy generation failed: {e}")
            return self._get_fallback_strategies(target)
    
    async def _execute_parallel_discovery(
        self, 
        target: DiscoveryTarget, 
        strategies: List[SearchStrategy],
        selected_methods: Optional[List[str]] = None
    ) -> Dict[str, List[ProviderCandidate]]:
        """
        Execute multiple discovery methods in parallel
        
        Args:
            target: Discovery target
            strategies: AI-generated search strategies
            selected_methods: Optional list of methods to execute
            
        Returns:
            Dictionary mapping method names to candidate lists
        """
        
        # Determine which methods to execute
        available_methods = {
            'search_engines': self._discover_via_search_engines,
            'regulatory_bodies': self._discover_via_regulatory_bodies,
            'competitor_analysis': self._discover_via_competitor_analysis,
            'social_intelligence': self._discover_via_social_intelligence,
            'industry_reports': self._discover_via_industry_reports,
            'news_analysis': self._discover_via_news_analysis
        }
        
        methods_to_execute = selected_methods or list(available_methods.keys())
        
        # Create discovery tasks
        discovery_tasks = {}
        for method_name in methods_to_execute:
            if method_name in available_methods:
                method_func = available_methods[method_name]
                task = self._execute_discovery_method_safely(
                    method_name, method_func, target, strategies
                )
                discovery_tasks[method_name] = task
        
        # Execute all discovery methods in parallel with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*discovery_tasks.values(), return_exceptions=True),
                timeout=300.0  # 5-minute timeout for all discovery methods
            )
        except asyncio.TimeoutError:
            self.logger.error("Discovery methods execution timed out")
            results = [[] for _ in discovery_tasks]  # Return empty results
        
        # Process results
        discovery_results = {}
        for i, (method_name, result) in enumerate(zip(discovery_tasks.keys(), results)):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Discovery method {method_name} failed: {result}")
                discovery_results[method_name] = []
            else:
                candidates = result if isinstance(result, list) else []
                discovery_results[method_name] = candidates
                self.logger.info(f"✅ {method_name}: {len(candidates)} candidates found")
        
        return discovery_results
    
    async def _execute_discovery_method_safely(
        self,
        method_name: str,
        method_func: callable,
        target: DiscoveryTarget,
        strategies: List[SearchStrategy]
    ) -> List[ProviderCandidate]:
        """
        Safely execute a discovery method with error handling and timeout
        """
        method_start_time = time.time()
        
        try:
            # Execute the discovery method with individual timeout
            candidates = await asyncio.wait_for(
                method_func(target, strategies),
                timeout=120.0  # 2-minute timeout per method
            )
            
            method_time = time.time() - method_start_time
            
            # Update method performance tracking
            if method_name not in self.discovery_stats['method_performance']:
                self.discovery_stats['method_performance'][method_name] = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'avg_candidates': 0.0,
                    'avg_time': 0.0
                }
            
            method_stats = self.discovery_stats['method_performance'][method_name]
            method_stats['total_executions'] += 1
            method_stats['successful_executions'] += 1
            
            # Update running averages
            prev_avg_candidates = method_stats['avg_candidates']
            prev_avg_time = method_stats['avg_time']
            executions = method_stats['successful_executions']
            
            method_stats['avg_candidates'] = (prev_avg_candidates * (executions - 1) + len(candidates)) / executions
            method_stats['avg_time'] = (prev_avg_time * (executions - 1) + method_time) / executions
            
            self.logger.info(f"🎯 {method_name} completed: {len(candidates)} candidates in {method_time:.1f}s")
            return candidates
            
        except asyncio.TimeoutError:
            self.logger.error(f"⏰ {method_name} timed out after 2 minutes")
            return []
        except Exception as e:
            method_time = time.time() - method_start_time
            self.logger.error(f"❌ {method_name} failed after {method_time:.1f}s: {e}")
            
            # Update failure statistics
            if method_name in self.discovery_stats['method_performance']:
                self.discovery_stats['method_performance'][method_name]['total_executions'] += 1
            
            return []
    
    @retry_with_backoff(max_retries=2, backoff_factor=1.0)
    async def _discover_via_search_engines(
        self, 
        target: DiscoveryTarget, 
        strategies: List[SearchStrategy]
    ) -> List[ProviderCandidate]:
        """
        Discover providers via search engines with AI-guided queries
        
        This method uses the SearchEngineAgent to execute search queries
        and validate results using AI analysis.
        """
        search_agent = self.specialized_agents.get('search_engine')
        if not search_agent:
            self.logger.warning("Search engine agent not available")
            return []
        
        # Find search engine strategies
        search_strategies = [s for s in strategies if s.method == 'search_engines']
        if not search_strategies:
            self.logger.warning("No search engine strategies available")
            return []
        
        return await search_agent.discover_providers(target, search_strategies)
    
    async def _discover_via_regulatory_bodies(
        self, 
        target: DiscoveryTarget, 
        strategies: List[SearchStrategy]
    ) -> List[ProviderCandidate]:
        """
        Discover providers via regulatory databases and official registries
        """
        regulatory_agent = self.specialized_agents.get('regulatory')
        if not regulatory_agent:
            self.logger.warning("Regulatory scanner not available")
            return []
        
        regulatory_strategies = [s for s in strategies if s.method == 'regulatory_bodies']
        return await regulatory_agent.scan_regulatory_databases(target, regulatory_strategies)
    
    async def _discover_via_competitor_analysis(
        self, 
        target: DiscoveryTarget, 
        strategies: List[SearchStrategy]
    ) -> List[ProviderCandidate]:
        """
        Discover providers via competitive intelligence and market analysis
        """
        competitor_agent = self.specialized_agents.get('competitor')
        if not competitor_agent:
            self.logger.warning("Competitor analyzer not available")
            return []
        
        competitor_strategies = [s for s in strategies if s.method == 'competitor_analysis']
        return await competitor_agent.analyze_competitive_landscape(target, competitor_strategies)
    
    async def _discover_via_social_intelligence(
        self, 
        target: DiscoveryTarget, 
        strategies: List[SearchStrategy]
    ) -> List[ProviderCandidate]:
        """
        Discover providers via social media intelligence gathering
        """
        social_agent = self.specialized_agents.get('social_intel')
        if not social_agent:
            self.logger.warning("Social intelligence agent not available")
            return []
        
        social_strategies = [s for s in strategies if s.method == 'social_intelligence']
        return await social_agent.gather_social_intelligence(target, social_strategies)
    
    async def _discover_via_industry_reports(
        self, 
        target: DiscoveryTarget, 
        strategies: List[SearchStrategy]
    ) -> List[ProviderCandidate]:
        """
        Discover providers via industry reports and market publications
        
        This method uses AI knowledge to identify providers commonly mentioned
        in industry reports and market analysis for the target service category.
        """
        report_strategies = [s for s in strategies if s.method == 'industry_reports']
        
        if not report_strategies:
            self.logger.info("No industry report strategies available")
            return []
        
        # Use AI knowledge for industry report analysis
        report_prompt = f"""
        Based on your knowledge of industry reports and market research for {target.service_category} in {target.country},
        identify key service providers that would typically be mentioned in industry analysis.
        
        Target Market Context:
        - Service Category: {target.service_category}
        - Country: {target.country}
        - Market Size: {target.market_size_estimate}
        - Language: {target.language}
        
        Focus Areas:
        1. **Market Leaders**: Established companies with significant market share
        2. **Regional Players**: Strong regional presence in {target.country}
        3. **Emerging Companies**: Notable new entrants or fast-growing firms
        4. **Specialized Providers**: Niche specialists with unique offerings
        5. **Government-Mentioned**: Companies cited in official reports
        
        For each provider, consider:
        - Market position and reputation
        - Service offerings and specializations
        - Geographic coverage within {target.country}
        - Recent industry recognition or mentions
        
        Return JSON array with provider information:
        [{{
          "name": "company_name",
          "website": "estimated_website_url",
          "market_position": "leader/regional/emerging/specialist",
          "services_focus": ["primary_services"],
          "geographic_coverage": ["regions_in_{target.country}"],
          "industry_recognition": ["awards", "certifications", "mentions"],
          "confidence_reasoning": "why_this_provider_is_notable",
          "estimated_market_share": "small/medium/large",
          "recent_developments": ["news", "expansions", "partnerships"]
        }}]
        
        Limit to 15 most notable providers to ensure quality over quantity.
        """
        
        try:
            response = await self.ask_ai(
                prompt=report_prompt,
                provider="ollama",  # Use free provider for knowledge-based analysis
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1500
            )
            
            providers_data = safe_json_parse(response, default=[])
            
            candidates = []
            for provider_data in providers_data:
                try:
                    # Calculate confidence score based on available data
                    confidence_factors = {
                        'has_website': bool(provider_data.get('website', '')),
                        'has_services': bool(provider_data.get('services_focus', [])),
                        'has_coverage': bool(provider_data.get('geographic_coverage', [])),
                        'has_recognition': bool(provider_data.get('industry_recognition', [])),
                        'market_position': provider_data.get('market_position', '') in ['leader', 'regional']
                    }
                    
                    base_confidence = 0.5  # Base confidence for AI knowledge
                    confidence_boost = sum(confidence_factors.values()) * 0.1
                    final_confidence = min(0.8, base_confidence + confidence_boost)  # Cap at 0.8 for knowledge-based
                    
                    candidate = ProviderCandidate(
                        name=provider_data.get('name', 'Unknown'),
                        website=provider_data.get('website', ''),
                        discovery_method='industry_reports_ai_knowledge',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position=provider_data.get('market_position', 'unknown'),
                        services_preview=provider_data.get('services_focus', []),
                        ai_analysis={
                            'source': 'industry_knowledge',
                            'reasoning': provider_data.get('confidence_reasoning', ''),
                            'market_share': provider_data.get('estimated_market_share', 'unknown'),
                            'recognition': provider_data.get('industry_recognition', []),
                            'recent_developments': provider_data.get('recent_developments', []),
                            'geographic_coverage': provider_data.get('geographic_coverage', [])
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create candidate from industry data: {e}")
            
            self.logger.info(f"Industry reports analysis: {len(candidates)} candidates identified")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Industry reports discovery failed: {e}")
            return []
    
    async def _discover_via_news_analysis(
        self, 
        target: DiscoveryTarget, 
        strategies: List[SearchStrategy]
    ) -> List[ProviderCandidate]:
        """
        Discover providers via news and announcement analysis
        
        This method uses AI to identify companies that have been mentioned in
        recent news coverage related to the target service category.
        """
        news_strategies = [s for s in strategies if s.method == 'news_analysis']
        
        if not news_strategies:
            self.logger.info("No news analysis strategies available")
            return []
        
        # Use AI for news-based provider discovery
        news_prompt = f"""
        Based on recent news and industry announcements in the {target.service_category} sector in {target.country},
        identify companies that have been mentioned in relevant news coverage over the past 12 months.
        
        Target Market Context:
        - Service Category: {target.service_category}
        - Country: {target.country}
        - Language: {target.language}
        - Market Context: {target.market_size_estimate}
        
        News Categories to Consider:
        1. **Business Announcements**: New service launches, expansions, partnerships
        2. **Industry Recognition**: Awards, certifications, rankings
        3. **Regulatory News**: Compliance updates, licensing changes
        4. **Market Developments**: Mergers, acquisitions, investments
        5. **Innovation News**: Technology adoption, digital transformation
        6. **Economic Impact**: Market reports, growth announcements
        
        For each newsworthy provider, analyze:
        - Reason for news coverage
        - Relevance to {target.service_category}
        - Market impact and significance
        - Credibility of news sources
        
        Return JSON array of newsworthy providers:
        [{{
          "name": "company_name",
          "website": "estimated_website_url", 
          "news_context": "reason_for_news_mention",
          "news_category": "announcement/recognition/regulatory/market/innovation/economic",
          "relevance_score": 0.0-1.0,
          "news_significance": "major/moderate/minor",
          "timeframe": "recent_months",
          "credibility_indicators": ["source_types", "verification_signals"],
          "business_impact": "description_of_impact",
          "market_implications": "what_this_means_for_market"
        }}]
        
        Focus on providers with substantial, credible news coverage. Limit to 12 providers.
        """
        
        try:
            response = await self.ask_ai(
                prompt=news_prompt,
                provider="ollama",  # Use free provider for knowledge-based analysis
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1200
            )
            
            news_data = safe_json_parse(response, default=[])
            
            candidates = []
            for item in news_data:
                try:
                    relevance = float(item.get('relevance_score', 0.5))
                    significance = item.get('news_significance', 'minor')
                    
                    # Calculate confidence based on news credibility and relevance
                    base_confidence = relevance * 0.6  # Start with relevance score
                    
                    # Boost confidence for significant news
                    if significance == 'major':
                        base_confidence += 0.2
                    elif significance == 'moderate':
                        base_confidence += 0.1
                    
                    # Consider credibility indicators
                    credibility_count = len(item.get('credibility_indicators', []))
                    base_confidence += min(0.1, credibility_count * 0.05)
                    
                    # Cap confidence for news-based discovery
                    final_confidence = min(0.75, base_confidence)
                    
                    candidate = ProviderCandidate(
                        name=item.get('name', 'Unknown'),
                        website=item.get('website', ''),
                        discovery_method='news_analysis_ai_knowledge',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='unknown',  # To be determined by further analysis
                        ai_analysis={
                            'source': 'news_analysis',
                            'news_context': item.get('news_context', ''),
                            'news_category': item.get('news_category', 'unknown'),
                            'significance': significance,
                            'timeframe': item.get('timeframe', 'recent'),
                            'credibility_indicators': item.get('credibility_indicators', []),
                            'business_impact': item.get('business_impact', ''),
                            'market_implications': item.get('market_implications', '')
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create candidate from news data: {e}")
            
            self.logger.info(f"News analysis: {len(candidates)} candidates identified")
            return candidates
            
        except Exception as e:
            self.logger.error(f"News analysis discovery failed: {e}")
            return []
    
    async def _consolidate_discovery_results(
        self, 
        discovery_results: Dict[str, List[ProviderCandidate]]
    ) -> List[ProviderCandidate]:
        """
        Consolidate and validate discovery results from all methods
        
        Args:
            discovery_results: Dictionary mapping method names to candidate lists
            
        Returns:
            Consolidated list of all candidates
        """
        all_candidates = []
        method_stats = {}
        
        for method_name, candidates in discovery_results.items():
            method_stats[method_name] = len(candidates)
            all_candidates.extend(candidates)
        
        self.logger.info(f"Consolidation results: {method_stats}")
        self.logger.info(f"Total candidates before deduplication: {len(all_candidates)}")
        
        return all_candidates
    
    async def _ai_deduplicate_and_rank(
        self, 
        candidates: List[ProviderCandidate], 
        target: DiscoveryTarget,
        max_candidates: int = 50
    ) -> List[ProviderCandidate]:
        """
        AI-powered deduplication and ranking of discovered candidates
        
        This method uses AI to identify duplicates (same company discovered through
        different methods) and rank all unique providers by quality and relevance.
        """
        if not candidates:
            return []
        
        self.logger.info(f"🔄 Starting AI deduplication and ranking of {len(candidates)} candidates")
        
        # For large candidate lists, process in batches to manage token limits
        batch_size = 20
        candidate_batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
        
        all_unique_providers = []
        
        for batch_idx, batch in enumerate(candidate_batches):
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(candidate_batches)} ({len(batch)} candidates)")
            
            # Create simplified candidate data for AI analysis
            batch_summary = []
            for i, candidate in enumerate(batch):
                batch_summary.append({
                    "id": i,
                    "name": candidate.name,
                    "website": candidate.website,
                    "discovery_method": candidate.discovery_method,
                    "confidence_score": candidate.confidence_score,
                    "market_position": candidate.market_position,
                    "services": candidate.services_preview[:3]  # Limit to top 3 services
                })
            
            dedup_prompt = f"""
            Analyze these discovered {target.service_category} providers in {target.country} for duplicates and rank by quality.
            
            Batch {batch_idx + 1} Candidates:
            {json.dumps(batch_summary, indent=2)}
            
            Target Market: {target.service_category} in {target.country}
            
            Tasks:
            1. **Duplicate Detection**: Identify candidates that represent the same company
               - Same company name (accounting for variations)
               - Same website domain
               - Similar service offerings and market position
            
            2. **Quality Assessment**: Rate each unique provider's legitimacy and relevance
               - Business legitimacy indicators
               - Service relevance to {target.service_category}
               - Market presence in {target.country}
               - Discovery method reliability
            
            3. **Ranking Criteria**: Consider these factors for ranking
               - Confidence score from discovery
               - Market position (leader > regional > niche > emerging)
               - Service offering completeness
               - Geographic relevance to {target.country}
            
            Return JSON with deduplicated and ranked results:
            {{
              "unique_providers": [{{
                "primary_candidate_id": 0,
                "best_name": "most_accurate_company_name",
                "best_website": "most_reliable_website_url",
                "duplicate_candidate_ids": [1, 2],
                "combined_confidence_score": 0.0-1.0,
                "market_position": "leader/regional/niche/emerging/unknown",
                "legitimacy_assessment": "high/medium/low",
                "relevance_score": 0.0-1.0,
                "ranking_score": 0.0-1.0,
                "quality_indicators": ["positive_signals"],
                "concerns": ["potential_issues"],
                "reasoning": "explanation_for_ranking"
              }}],
              "duplicates_found": 0,
              "ranking_methodology": "explanation_of_approach"
            }}
            """
            
            try:
                # Use Groq for faster complex analysis
                response = await self.ask_ai(
                    prompt=dedup_prompt,
                    provider="groq",
                    task_complexity=TaskComplexity.COMPLEX,
                    max_tokens=1500
                )
                
                dedup_data = safe_json_parse(response, default={"unique_providers": []})
                unique_providers = dedup_data.get('unique_providers', [])
                
                # Process unique providers from this batch
                for provider_data in unique_providers:
                    try:
                        primary_id = provider_data.get('primary_candidate_id')
                        if primary_id is not None and primary_id < len(batch):
                            base_candidate = batch[primary_id]
                            
                            # Create enhanced candidate with AI analysis
                            enhanced_candidate = ProviderCandidate(
                                name=provider_data.get('best_name', base_candidate.name),
                                website=provider_data.get('best_website', base_candidate.website),
                                discovery_method=f"ai_enhanced_{base_candidate.discovery_method}",
                                confidence_score=provider_data.get('combined_confidence_score', base_candidate.confidence_score),
                                business_category=target.service_category,
                                market_position=provider_data.get('market_position', base_candidate.market_position),
                                contact_info=base_candidate.contact_info,
                                services_preview=base_candidate.services_preview,
                                ai_analysis={
                                    **base_candidate.ai_analysis,
                                    'deduplication_analysis': provider_data,
                                    'ranking_score': provider_data.get('ranking_score', 0.5),
                                    'legitimacy_assessment': provider_data.get('legitimacy_assessment', 'medium'),
                                    'relevance_score': provider_data.get('relevance_score', 0.5),
                                    'quality_indicators': provider_data.get('quality_indicators', []),
                                    'concerns': provider_data.get('concerns', []),
                                    'batch_processed': batch_idx + 1
                                }
                            )
                            
                            all_unique_providers.append(enhanced_candidate)
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to create enhanced candidate: {e}")
                
                duplicates_found = dedup_data.get('duplicates_found', 0)
                self.logger.info(f"Batch {batch_idx + 1}: {len(unique_providers)} unique, {duplicates_found} duplicates")
                
            except Exception as e:
                self.logger.error(f"Deduplication failed for batch {batch_idx + 1}: {e}")
                # Fallback: add all candidates from this batch without deduplication
                for candidate in batch:
                    candidate.ai_analysis['deduplication_status'] = 'failed_fallback'
                    all_unique_providers.append(candidate)
        
        # Final ranking across all batches
        all_unique_providers.sort(
            key=lambda c: c.ai_analysis.get('ranking_score', 0.5), 
            reverse=True
        )
        
        # Apply final limit
        final_candidates = all_unique_providers[:max_candidates]
        
        self.logger.info(
            f"✅ Deduplication complete: {len(all_unique_providers)} unique providers, "
            f"returning top {len(final_candidates)}"
        )
        
        return final_candidates
    
    async def _enhance_strategies_with_memory(
        self, 
        strategies: List[SearchStrategy], 
        target: DiscoveryTarget
    ) -> List[SearchStrategy]:
        """
        Enhance search strategies with learnings from previous successful discoveries
        
        This method applies machine learning patterns from previous discoveries
        to improve the current search strategies.
        """
        market_key = f"{target.country}_{target.service_category}"
        
        # Check for similar previous discoveries
        similar_patterns = await self._get_similar_discovery_patterns(target)
        
        if not similar_patterns:
            self.logger.info("No similar discovery patterns found, using base strategies")
            return strategies
        
        self.logger.info(f"📚 Applying learned patterns from {len(similar_patterns)} similar discoveries")
        
        # AI-powered strategy enhancement
        enhancement_prompt = f"""
        Enhance these search strategies based on patterns from previous successful discoveries:
        
        Current Strategies for {target.service_category} in {target.country}:
        {json.dumps([{
            'method': s.method,
            'priority': s.priority,
            'queries': s.queries[:3],  # Show only first 3 queries
            'platforms': s.platforms[:3]  # Show only first 3 platforms
        } for s in strategies], indent=2)}
        
        Previous Successful Discovery Patterns:
        {json.dumps(similar_patterns[-3:], indent=2)}  # Last 3 similar discoveries
        
        Enhancement Guidelines:
        1. **Query Optimization**: Add successful search terms from similar markets
        2. **Platform Prioritization**: Boost platforms that performed well previously
        3. **Method Weighting**: Increase priority for methods with high success rates
        4. **Regional Adaptation**: Apply location-specific optimizations
        
        Return enhanced strategies with improvements:
        {{
          "enhanced_strategies": [{{
            "method": "method_name",
            "original_priority": 1-10,
            "enhanced_priority": 1-10,
            "additional_queries": ["new_query_terms"],
            "additional_platforms": ["new_platforms"],
            "optimization_reasoning": "why_this_enhancement"
          }}],
          "enhancement_summary": "overall_improvements_made"
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=enhancement_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1000
            )
            
            enhancement_data = safe_json_parse(response, default={"enhanced_strategies": []})
            enhanced_strategy_data = enhancement_data.get('enhanced_strategies', [])
            
            # Apply enhancements to original strategies
            enhanced_strategies = []
            for i, strategy in enumerate(strategies):
                enhanced_strategy = strategy
                
                # Find corresponding enhancement data
                for enhancement in enhanced_strategy_data:
                    if enhancement.get('method') == strategy.method:
                        # Apply enhancements
                        enhanced_strategy.priority = enhancement.get('enhanced_priority', strategy.priority)
                        enhanced_strategy.queries.extend(enhancement.get('additional_queries', []))
                        enhanced_strategy.platforms.extend(enhancement.get('additional_platforms', []))
                        
                        # Add enhancement metadata
                        enhanced_strategy.metadata = {
                            **strategy.metadata,
                            'enhancement_applied': True,
                            'optimization_reasoning': enhancement.get('optimization_reasoning', ''),
                            'original_priority': enhancement.get('original_priority', strategy.priority)
                        }
                        break
                
                enhanced_strategies.append(enhanced_strategy)
            
            self.logger.info(f"✅ Enhanced {len(enhanced_strategies)} strategies with learning patterns")
            return enhanced_strategies
            
        except Exception as e:
            self.logger.warning(f"Strategy enhancement failed: {e}")
            return strategies
    
    def _get_fallback_strategies(self, target: DiscoveryTarget) -> List[SearchStrategy]:
        """
        Get basic fallback search strategies when AI generation fails
        
        These are manually crafted, reliable strategies that should work
        for most service discovery scenarios.
        """
        fallback_strategies = [
            SearchStrategy(
                method="search_engines",
                priority=9,
                queries=[
                    f"{target.service_category} providers {target.country}",
                    f"{target.service_category} companies {target.country}",
                    f"best {target.service_category} {target.country}",
                    f"top {target.service_category} services {target.country}",
                    f"{target.service_category} directory {target.country}"
                ],
                platforms=["google", "bing"],
                expected_yield="15-25",
                ai_analysis_needed=True,
                follow_up_actions=["validate_websites", "extract_contact_info"],
                metadata={"fallback": True, "reliability": "high"}
            ),
            SearchStrategy(
                method="regulatory_bodies",
                priority=8,
                queries=[
                    f"licensed {target.service_category} {target.country}",
                    f"registered {target.service_category} providers {target.country}"
                ],
                platforms=target.regulatory_bodies,
                expected_yield="5-15",
                ai_analysis_needed=True,
                follow_up_actions=["verify_licenses", "check_status"],
                metadata={"fallback": True, "reliability": "high"}
            ),
            SearchStrategy(
                method="industry_reports",
                priority=7,
                queries=[
                    f"{target.service_category} market leaders {target.country}",
                    f"major {target.service_category} companies {target.country}"
                ],
                platforms=["industry_knowledge"],
                expected_yield="8-12",
                ai_analysis_needed=True,
                follow_up_actions=["verify_market_position"],
                metadata={"fallback": True, "reliability": "medium"}
            )
        ]
        
        self.logger.info(f"🔄 Using {len(fallback_strategies)} fallback strategies")
        return fallback_strategies
    
    async def _fallback_discovery(
        self, 
        target: DiscoveryTarget, 
        max_candidates: int = 20
    ) -> List[ProviderCandidate]:
        """
        Basic fallback discovery when main discovery process fails
        
        This provides a simple but reliable discovery mechanism that should
        always return some results, even if the full AI-powered discovery fails.
        """
        self.logger.info("🔄 Executing fallback discovery")
        
        try:
            # Simple AI-based provider discovery
            fallback_prompt = f"""
            List the most well-known {target.service_category} providers in {target.country}.
            
            Focus on:
            1. Major market leaders
            2. Well-established regional players
            3. Government-recognized providers
            
            Return JSON array:
            [{{
              "name": "company_name",
              "estimated_website": "website_url",
              "market_position": "leader/regional",
              "known_for": "primary_services"
            }}]
            
            Limit to 10 most notable providers.
            """
            
            response = await self.ask_ai(
                prompt=fallback_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.SIMPLE,
                max_tokens=800
            )
            
            fallback_data = safe_json_parse(response, default=[])
            
            candidates = []
            for provider_data in fallback_data:
                try:
                    candidate = ProviderCandidate(
                        name=provider_data.get('name', 'Unknown'),
                        website=provider_data.get('estimated_website', ''),
                        discovery_method='fallback_ai_knowledge',
                        confidence_score=0.4,  # Lower confidence for fallback
                        business_category=target.service_category,
                        market_position=provider_data.get('market_position', 'unknown'),
                        ai_analysis={
                            'source': 'fallback_discovery',
                            'known_for': provider_data.get('known_for', ''),
                            'fallback_reason': 'main_discovery_failed'
                        }
                    )
                    candidates.append(candidate)
                except Exception as e:
                    self.logger.warning(f"Failed to create fallback candidate: {e}")
            
            return candidates[:max_candidates]
            
        except Exception as e:
            self.logger.error(f"Fallback discovery also failed: {e}")
            return []
    
    async def _update_discovery_patterns(
        self, 
        target: DiscoveryTarget, 
        candidates: List[ProviderCandidate],
        strategies: List[SearchStrategy],
        discovery_time: float
    ):
        """
        Update learning patterns based on discovery results
        
        This method stores successful discovery patterns for future use,
        implementing a simple machine learning mechanism.
        """
        market_key = f"{target.country}_{target.service_category}"
        
        # Analyze which methods were most successful
        method_success = {}
        for candidate in candidates:
            method = candidate.discovery_method.split('_')[0]  # Get base method name
            method_success[method] = method_success.get(method, 0) + 1
        
        # Create discovery pattern record
        pattern_record = {
            'timestamp': datetime.now().isoformat(),
            'target': {
                'country': target.country,
                'service_category': target.service_category,
                'language': target.language,
                'discovery_depth': target.discovery_depth
            },
            'results': {
                'total_candidates': len(candidates),
                'discovery_time_seconds': discovery_time,
                'method_success': method_success,
                'avg_confidence': sum(c.confidence_score for c in candidates) / len(candidates) if candidates else 0.0
            },
            'strategies_used': len(strategies),
            'success_indicators': {
                'found_candidates': len(candidates) > 0,
                'reasonable_time': discovery_time < 300,  # 5 minutes
                'good_yield': len(candidates) >= 5,
                'high_confidence': any(c.confidence_score > 0.8 for c in candidates)
            }
        }
        
        # Store in memory cache
        if market_key not in self.discovery_patterns_cache:
            self.discovery_patterns_cache[market_key] = []
        
        self.discovery_patterns_cache[market_key].append(pattern_record)
        
        # Keep only last 20 patterns per market
        if len(self.discovery_patterns_cache[market_key]) > 20:
            self.discovery_patterns_cache[market_key] = self.discovery_patterns_cache[market_key][-20:]
        
        # Store successful strategies separately
        if pattern_record['success_indicators']['found_candidates']:
            self.successful_strategies_cache[market_key] = {
                'last_successful': pattern_record,
                'method_performance': method_success,
                'best_strategies': [s.method for s in strategies if s.priority > 7]
            }
        
        # Save to Redis if available
        await self._save_discovery_patterns()
        
        self.logger.info(f"📚 Updated discovery patterns for {market_key}")
    
    def _update_performance_stats(self, candidates_found: int, discovery_time: float):
        """Update overall performance statistics"""
        self.discovery_stats['total_discoveries'] += 1
        
        if candidates_found > 0:
            self.discovery_stats['successful_discoveries'] += 1
        
        # Update running averages
        total = self.discovery_stats['total_discoveries']
        
        prev_avg_time = self.discovery_stats['avg_discovery_time']
        self.discovery_stats['avg_discovery_time'] = (prev_avg_time * (total - 1) + discovery_time) / total
        
        prev_avg_candidates = self.discovery_stats['avg_candidates_found']
        self.discovery_stats['avg_candidates_found'] = (prev_avg_candidates * (total - 1) + candidates_found) / total
    
    async def _get_cached_strategies(self, cache_key: str) -> Optional[List[SearchStrategy]]:
        """Retrieve cached search strategies"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(f"strategies:{cache_key}")
            if cached_data:
                strategies_data = json.loads(cached_data)
                return [SearchStrategy(**data) for data in strategies_data]
        except Exception as e:
            self.logger.warning(f"Failed to get cached strategies: {e}")
        
        return None
    
    async def _cache_strategies(self, cache_key: str, strategies: List[SearchStrategy]):
        """Cache search strategies for reuse"""
        if not self.redis_client:
            return
        
        try:
            strategies_data = [asdict(strategy) for strategy in strategies]
            await self.redis_client.setex(
                f"strategies:{cache_key}",
                self.config.cache_ttl,
                json.dumps(strategies_data)
            )
        except Exception as e:
            self.logger.warning(f"Failed to cache strategies: {e}")
    
    async def _get_similar_discovery_patterns(self, target: DiscoveryTarget) -> List[Dict]:
        """Get discovery patterns from similar markets"""
        similar_patterns = []
        
        # Check in-memory cache first
        for market_key, patterns in self.discovery_patterns_cache.items():
            country, category = market_key.split('_', 1)
            
            # Match by country or category
            if country == target.country or category == target.service_category:
                similar_patterns.extend(patterns[-3:])  # Last 3 patterns from each similar market
        
        # Try to load from Redis if available
        if not similar_patterns and self.redis_client:
            try:
                cached_patterns = await self.redis_client.get(f"patterns:{target.country}_{target.service_category}")
                if cached_patterns:
                    similar_patterns = json.loads(cached_patterns)
            except Exception as e:
                self.logger.warning(f"Failed to load patterns from Redis: {e}")
        
        return similar_patterns
    
    async def _load_discovery_patterns(self):
        """Load discovery patterns from persistent storage"""
        if not self.redis_client:
            return
        
        try:
            # Load discovery patterns cache
            patterns_data = await self.redis_client.get("discovery_patterns_cache")
            if patterns_data:
                self.discovery_patterns_cache = json.loads(patterns_data)
            
            # Load successful strategies cache
            strategies_data = await self.redis_client.get("successful_strategies_cache")
            if strategies_data:
                self.successful_strategies_cache = json.loads(strategies_data)
            
            self.logger.info("📚 Loaded discovery patterns from cache")
            
        except Exception as e:
            self.logger.warning(f"Failed to load discovery patterns: {e}")
    
    async def _save_discovery_patterns(self):
        """Save discovery patterns to persistent storage"""
        if not self.redis_client:
            return
        
        try:
            # Save discovery patterns cache
            await self.redis_client.setex(
                "discovery_patterns_cache",
                86400,  # 24 hours
                json.dumps(self.discovery_patterns_cache)
            )
            
            # Save successful strategies cache
            await self.redis_client.setex(
                "successful_strategies_cache",
                86400,  # 24 hours
                json.dumps(self.successful_strategies_cache)
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to save discovery patterns: {e}")
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get comprehensive discovery statistics"""
        base_stats = self.get_metrics()
        
        discovery_stats = {
            **base_stats,
            "discovery_performance": self.discovery_stats,
            "active_agents": list(self.specialized_agents.keys()),
            "cached_patterns": {
                market: len(patterns) for market, patterns in self.discovery_patterns_cache.items()
            },
            "successful_strategies": list(self.successful_strategies_cache.keys())
        }
        
        return discovery_stats