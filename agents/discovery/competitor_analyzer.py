# agents/discovery/competitor_analyzer.py
"""
Competitor Analysis Agent

This agent specializes in discovering service providers through competitive
intelligence, market analysis, and relationship mapping. It uses AI to understand
competitive landscapes and identify providers through competitor networks.
"""

import asyncio
import logging
import json
import time
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup
import hashlib

from ..base import BaseAgent, AgentConfig, AgentError, safe_json_parse, retry_with_backoff, rate_limiter
from .models import DiscoveryTarget, ProviderCandidate, SearchStrategy
from services.ai_async_client import AIAsyncClient, TaskComplexity

logger = logging.getLogger(__name__)


class CompetitorAnalyzer(BaseAgent):
    """
    Specialized agent for competitive intelligence and market analysis
    
    Features:
    - AI-powered competitive landscape mapping
    - Partner and supplier network analysis
    - Market leader identification and analysis
    - SEO competitor discovery
    - Investment and acquisition tracking
    - Industry ecosystem mapping
    """
    
    def __init__(self, ai_client: AIAsyncClient, redis_client=None):
        config = AgentConfig(
            name="CompetitorAnalyzer",
            max_retries=2,
            rate_limit=15,  # Conservative for competitive analysis
            preferred_ai_provider="ollama",  # Use free provider for analysis
            task_complexity=TaskComplexity.COMPLEX,
            cache_ttl=3600,  # 1 hour cache for competitive data
            debug=False
        )
        
        super().__init__(config, ai_client)
        
        # Redis client for caching competitive intelligence
        self.redis_client = redis_client
        
        # HTTP session for competitive research
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Competitive intelligence sources
        self.intelligence_sources = {
            'industry_reports': {
                'crunchbase': 'https://www.crunchbase.com',
                'pitchbook': 'https://pitchbook.com',
                'cbinsights': 'https://www.cbinsights.com'
            },
            'market_research': {
                'gartner': 'https://www.gartner.com',
                'forrester': 'https://www.forrester.com',
                'idc': 'https://www.idc.com'
            },
            'financial_data': {
                'bloomberg': 'https://www.bloomberg.com',
                'reuters': 'https://www.reuters.com',
                'yahoo_finance': 'https://finance.yahoo.com'
            }
        }
        
        # Competitive analysis cache
        self.competitive_cache = {}
        
        # Performance tracking
        self.analysis_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'competitors_mapped': 0,
            'networks_analyzed': 0,
            'method_performance': {}
        }
    
    async def _setup_agent(self) -> None:
        """Initialize HTTP session and competitive intelligence configurations"""
        try:
            # Setup HTTP session for competitive research
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(
                limit=8,
                limit_per_host=4,
                ttl_dns_cache=300
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Market Research Bot/1.0 (Competitive Intelligence)',
                    'Accept': 'text/html,application/json,*/*',
                    'Accept-Language': 'en-US,en;q=0.9'
                }
            )
            
            # Load competitive cache
            await self._load_competitive_cache()
            
            self.logger.info("Competitor analyzer initialized successfully")
            
        except Exception as e:
            raise AgentError(self.config.name, f"Failed to initialize competitor analyzer: {e}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup HTTP session and save cache"""
        try:
            # Save competitive cache
            await self._save_competitive_cache()
            
            # Close HTTP session
            if self.session and not self.session.closed:
                await self.session.close()
                await asyncio.sleep(0.1)
            
            self.logger.info("Competitor analyzer cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup warning for competitor analyzer: {e}")
    
    async def analyze_competitive_landscape(
        self, 
        target: DiscoveryTarget,
        strategies: List[SearchStrategy]
    ) -> List[ProviderCandidate]:
        """
        Main competitive analysis method for discovering providers
        
        Args:
            target: Discovery target specification
            strategies: Competitive analysis strategies
            
        Returns:
            List of providers discovered through competitive intelligence
        """
        self.logger.info(f"🔍 Starting competitive analysis for {target.service_category} in {target.country}")
        
        start_time = time.time()
        all_candidates = []
        
        try:
            # Filter for competitor analysis strategies
            competitor_strategies = [s for s in strategies if s.method == 'competitor_analysis']
            
            if not competitor_strategies:
                # Generate default competitive strategy
                competitor_strategies = await self._generate_default_competitive_strategy(target)
            
            # Phase 1: Identify market leaders and seed companies
            market_leaders = await self._identify_market_leaders(target)
            
            # Phase 2: Analyze competitive ecosystem
            competitive_ecosystem = await self._map_competitive_ecosystem(target, market_leaders)
            
            # Phase 3: Execute competitive analysis strategies
            for strategy in competitor_strategies:
                strategy_candidates = await self._execute_competitive_strategy(
                    target, strategy, competitive_ecosystem
                )
                all_candidates.extend(strategy_candidates)
                
                # Rate limiting between strategies
                await asyncio.sleep(2.0)
            
            # Phase 4: Analyze competitor networks and partnerships
            network_candidates = await self._analyze_competitor_networks(
                market_leaders, target, competitive_ecosystem
            )
            all_candidates.extend(network_candidates)
            
            # Phase 5: Validate and enhance candidates
            validated_candidates = await self._validate_competitive_candidates(
                all_candidates, target, competitive_ecosystem
            )
            
            analysis_time = time.time() - start_time
            self.analysis_stats['total_analyses'] += 1
            if validated_candidates:
                self.analysis_stats['successful_analyses'] += 1
                self.analysis_stats['competitors_mapped'] += len(validated_candidates)
            
            self.logger.info(
                f"✅ Competitive analysis complete: {len(validated_candidates)} candidates "
                f"found in {analysis_time:.1f}s"
            )
            
            return validated_candidates
            
        except Exception as e:
            self.logger.error(f"❌ Competitive analysis failed: {e}")
            raise AgentError(self.config.name, f"Competitive analysis failed: {e}")
    
    async def _identify_market_leaders(self, target: DiscoveryTarget) -> List[Dict[str, Any]]:
        """
        Use AI to identify market leaders in the target service category
        
        Args:
            target: Discovery target with market context
            
        Returns:
            List of market leader information
        """
        
        leaders_prompt = f"""
        Identify the major market leaders and key players in {target.service_category} industry in {target.country}.
        
        Target Market Context:
        - Country: {target.country}
        - Service Category: {target.service_category}
        - Language: {target.language}
        - Market Size: {target.market_size_estimate}
        
        Identify different tiers of companies:
        
        1. **Global Leaders**: International companies with presence in {target.country}
        2. **Regional Champions**: Strong regional players dominating {target.country}
        3. **Local Leaders**: Domestic companies with significant market share
        4. **Emerging Players**: Fast-growing companies gaining market share
        5. **Niche Specialists**: Companies with specialized {target.service_category} offerings
        
        For each company, provide:
        - Market position and estimated market share
        - Key competitive advantages
        - Geographic coverage within {target.country}
        - Notable partnerships or acquisitions
        - Investment backing or funding status
        
        Return JSON with market leaders:
        {{
          "market_leaders": [{{
            "company_name": "official_company_name",
            "tier": "global/regional/local/emerging/niche",
            "estimated_market_share": "percentage_or_description",
            "website": "primary_website_url",
            "headquarters": "location",
            "geographic_coverage": ["regions_in_{target.country}"],
            "key_strengths": ["competitive_advantages"],
            "recent_developments": ["acquisitions", "partnerships", "funding"],
            "competitor_relationships": ["partners", "competitors", "suppliers"],
            "market_positioning": "how_they_compete",
            "customer_segments": ["target_customer_types"],
            "confidence_level": 0.0-1.0
          }}],
          "market_dynamics": {{
            "market_concentration": "high/medium/low",
            "key_trends": ["industry_trends"],
            "competitive_factors": ["what_drives_competition"],
            "entry_barriers": ["barriers_to_entry"]
          }}
        }}
        
        Focus on 12-15 most significant players across all tiers.
        """
        
        try:
            response = await self.ask_ai(
                prompt=leaders_prompt,
                provider="ollama",  # Use free provider for market analysis
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2000
            )
            
            leaders_data = safe_json_parse(response, default={"market_leaders": []})
            market_leaders = leaders_data.get('market_leaders', [])
            
            self.logger.info(f"📊 Market leaders identified: {len(market_leaders)} companies across all tiers")
            return market_leaders
            
        except Exception as e:
            self.logger.error(f"Market leader identification failed: {e}")
            return self._get_fallback_market_leaders(target)
    
    async def _map_competitive_ecosystem(
        self, 
        target: DiscoveryTarget,
        market_leaders: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a comprehensive map of the competitive ecosystem
        
        Args:
            target: Discovery target
            market_leaders: Identified market leaders
            
        Returns:
            Competitive ecosystem mapping
        """
        
        ecosystem_prompt = f"""
        Map the competitive ecosystem for {target.service_category} in {target.country} based on these market leaders:
        
        Market Leaders:
        {json.dumps([{
            'name': leader.get('company_name', ''),
            'tier': leader.get('tier', ''),
            'strengths': leader.get('key_strengths', [])
        } for leader in market_leaders[:10]], indent=2)}
        
        Create comprehensive ecosystem mapping:
        
        1. **Competitive Clusters**: Groups of companies that compete directly
        2. **Partnership Networks**: Companies that frequently partner together
        3. **Supply Chain Relationships**: Supplier and vendor relationships
        4. **Investment Connections**: Shared investors, parent companies, subsidiaries
        5. **Technology Ecosystems**: Companies using similar technologies or platforms
        6. **Geographic Segments**: Regional competition patterns within {target.country}
        
        Identify relationship patterns:
        - Who competes with whom in which segments
        - Who partners with whom for what purposes
        - Who acquires or invests in similar companies
        - Who serves similar customer segments
        
        Return JSON ecosystem map:
        {{
          "competitive_clusters": [{{
            "cluster_name": "descriptive_name",
            "companies": ["company1", "company2"],
            "competition_focus": "what_they_compete_on",
            "market_segment": "customer_segment_or_geographic_area"
          }}],
          "partnership_networks": [{{
            "network_type": "technology/channel/strategic",
            "core_companies": ["main_partners"],
            "ecosystem_companies": ["extended_partners"],
            "partnership_purpose": "why_they_partner"
          }}],
          "acquisition_patterns": [{{
            "acquiring_companies": ["active_acquirers"],
            "acquisition_targets": ["types_of_companies_acquired"],
            "acquisition_rationale": ["strategic_reasons"]
          }}],
          "market_gaps": [{{
            "gap_description": "underserved_market_segment",
            "potential_players": ["companies_that_might_fill_gap"],
            "opportunity_size": "market_potential"
          }}],
          "ecosystem_insights": {{
            "dominant_business_models": ["primary_business_models"],
            "key_differentiation_factors": ["how_companies_differentiate"],
            "emerging_competitive_threats": ["new_types_of_competition"]
          }}
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=ecosystem_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=1800
            )
            
            ecosystem_data = safe_json_parse(response, default={})
            
            # Add market leaders to ecosystem data
            ecosystem_data['market_leaders'] = market_leaders
            
            self.logger.info("🗺️ Competitive ecosystem mapped successfully")
            return ecosystem_data
            
        except Exception as e:
            self.logger.error(f"Ecosystem mapping failed: {e}")
            return {'market_leaders': market_leaders}
    
    async def _execute_competitive_strategy(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        ecosystem: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Execute a specific competitive analysis strategy
        
        Args:
            target: Discovery target
            strategy: Competitive analysis strategy
            ecosystem: Competitive ecosystem mapping
            
        Returns:
            List of candidates from competitive analysis
        """
        self.logger.info(f"🎯 Executing competitive strategy: {strategy.method}")
        
        candidates = []
        
        # Execute different competitive analysis methods
        analysis_methods = {
            'market_leader_analysis': self._analyze_market_leaders,
            'partnership_discovery': self._discover_through_partnerships,
            'acquisition_analysis': self._analyze_acquisitions,
            'supplier_network_analysis': self._analyze_supplier_networks,
            'customer_reference_analysis': self._analyze_customer_references,
            'seo_competitor_analysis': self._analyze_seo_competitors
        }
        
        # Execute multiple analysis methods
        for method_name, method_func in analysis_methods.items():
            try:
                method_candidates = await method_func(target, strategy, ecosystem)
                candidates.extend(method_candidates)
                
                # Track method performance
                if method_name not in self.analysis_stats['method_performance']:
                    self.analysis_stats['method_performance'][method_name] = {
                        'executions': 0,
                        'avg_results': 0.0
                    }
                
                method_stats = self.analysis_stats['method_performance'][method_name]
                method_stats['executions'] += 1
                prev_avg = method_stats['avg_results']
                executions = method_stats['executions']
                method_stats['avg_results'] = (prev_avg * (executions - 1) + len(method_candidates)) / executions
                
                # Rate limiting between methods
                await asyncio.sleep(1.5)
                
            except Exception as e:
                self.logger.error(f"Competitive method {method_name} failed: {e}")
                continue
        
        self.logger.info(f"📋 Competitive strategy executed: {len(candidates)} candidates found")
        return candidates
    
    async def _analyze_market_leaders(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        ecosystem: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Discover providers by analyzing market leaders and their ecosystems
        
        Args:
            target: Discovery target
            strategy: Search strategy
            ecosystem: Competitive ecosystem
            
        Returns:
            List of candidates from market leader analysis
        """
        candidates = []
        market_leaders = ecosystem.get('market_leaders', [])
        
        for leader in market_leaders:
            try:
                # Convert market leader to candidate
                confidence_level = leader.get('confidence_level', 0.7)
                
                candidate = ProviderCandidate(
                    name=leader.get('company_name', 'Unknown'),
                    website=leader.get('website', ''),
                    discovery_method='competitive_market_leader_analysis',
                    confidence_score=confidence_level,
                    business_category=target.service_category,
                    market_position=leader.get('tier', 'unknown'),
                    ai_analysis={
                        'source': 'market_leader_analysis',
                        'market_tier': leader.get('tier'),
                        'market_share': leader.get('estimated_market_share', 'unknown'),
                        'key_strengths': leader.get('key_strengths', []),
                        'geographic_coverage': leader.get('geographic_coverage', []),
                        'recent_developments': leader.get('recent_developments', []),
                        'competitor_relationships': leader.get('competitor_relationships', []),
                        'market_positioning': leader.get('market_positioning', ''),
                        'customer_segments': leader.get('customer_segments', [])
                    }
                )
                
                candidates.append(candidate)
                
            except Exception as e:
                self.logger.warning(f"Failed to process market leader: {e}")
                continue
        
        return candidates
    
    async def _discover_through_partnerships(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        ecosystem: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Discover providers through partnership network analysis
        
        Args:
            target: Discovery target
            strategy: Search strategy
            ecosystem: Competitive ecosystem
            
        Returns:
            List of candidates discovered through partnerships
        """
        
        partnership_prompt = f"""
        Analyze partnership networks in {target.service_category} industry in {target.country} to discover additional providers.
        
        Known Market Players:
        {json.dumps([leader.get('company_name', '') for leader in ecosystem.get('market_leaders', [])[:8]], indent=2)}
        
        Partnership Networks:
        {json.dumps(ecosystem.get('partnership_networks', []), indent=2)}
        
        Discover providers through partnership analysis:
        
        1. **Technology Partners**: Companies that integrate or partner with known leaders
        2. **Channel Partners**: Resellers, distributors, implementation partners
        3. **Strategic Alliances**: Joint venture partners, co-marketing partners
        4. **Ecosystem Partners**: Platform partners, marketplace participants
        5. **Supply Chain Partners**: Suppliers, vendors, service providers
        
        For each partnership type, identify companies that:
        - Serve similar customers in {target.country}
        - Offer complementary {target.service_category} services
        - Have announced partnerships with known players
        - Participate in industry ecosystems
        
        Return JSON with partnership-discovered providers:
        [{{
          "company_name": "partner_company_name",
          "website": "estimated_website",
          "partnership_type": "technology/channel/strategic/ecosystem/supply",
          "partner_companies": ["companies_they_partner_with"],
          "partnership_purpose": "reason_for_partnership",
          "market_role": "their_role_in_ecosystem",
          "service_focus": ["their_primary_services"],
          "discovery_reasoning": "why_this_partnership_indicates_relevance"
        }}]
        
        Focus on 10-12 most relevant partnership-discovered providers.
        """
        
        try:
            response = await self.ask_ai(
                prompt=partnership_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1200
            )
            
            partnership_data = safe_json_parse(response, default=[])
            
            candidates = []
            for partner_data in partnership_data:
                try:
                    # Calculate confidence based on partnership strength
                    partnership_type = partner_data.get('partnership_type', 'unknown')
                    partner_count = len(partner_data.get('partner_companies', []))
                    
                    base_confidence = 0.6
                    
                    # Boost confidence for strategic partnerships
                    if partnership_type in ['strategic', 'technology']:
                        base_confidence += 0.1
                    
                    # Boost for multiple partnerships
                    partnership_boost = min(0.2, partner_count * 0.05)
                    final_confidence = min(0.8, base_confidence + partnership_boost)
                    
                    candidate = ProviderCandidate(
                        name=partner_data.get('company_name', 'Unknown'),
                        website=partner_data.get('website', ''),
                        discovery_method='competitive_partnership_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='partner_ecosystem',
                        ai_analysis={
                            'source': 'partnership_analysis',
                            'partnership_type': partnership_type,
                            'partner_companies': partner_data.get('partner_companies', []),
                            'partnership_purpose': partner_data.get('partnership_purpose', ''),
                            'market_role': partner_data.get('market_role', ''),
                            'service_focus': partner_data.get('service_focus', []),
                            'discovery_reasoning': partner_data.get('discovery_reasoning', '')
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create partnership candidate: {e}")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Partnership analysis failed: {e}")
            return []
    
    async def _analyze_acquisitions(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        ecosystem: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Discover providers through acquisition and investment analysis
        
        Args:
            target: Discovery target
            strategy: Search strategy
            ecosystem: Competitive ecosystem
            
        Returns:
            List of candidates discovered through acquisition analysis
        """
        
        acquisition_prompt = f"""
        Analyze acquisition and investment patterns in {target.service_category} industry in {target.country}.
        
        Known Market Leaders:
        {json.dumps([leader.get('company_name', '') for leader in ecosystem.get('market_leaders', [])[:6]], indent=2)}
        
        Acquisition Patterns:
        {json.dumps(ecosystem.get('acquisition_patterns', []), indent=2)}
        
        Discover providers through acquisition and investment analysis:
        
        1. **Recent Acquisitions**: Companies acquired by market leaders in past 2-3 years
        2. **Investment Targets**: Companies that received funding for {target.service_category}
        3. **Merger Activity**: Companies involved in mergers or joint ventures
        4. **Spin-offs**: New companies created from larger organizations
        5. **Portfolio Companies**: Companies owned by known investors or holding companies
        
        Focus on discovering:
        - Target companies that were acquired but may still operate independently
        - Funded startups that are growing in {target.service_category}
        - Companies that emerged from corporate restructuring
        - Portfolio companies of known industry investors
        
        Return JSON with acquisition-discovered providers:
        [{{
          "company_name": "target_company_name",
          "website": "company_website",
          "acquisition_type": "acquired/funded/merged/spinoff/portfolio",
          "acquiring_company": "parent_or_acquiring_company",
          "transaction_date": "approximate_date_or_year",
          "transaction_value": "funding_amount_if_known",
          "current_status": "independent/integrated/subsidiary",
          "strategic_rationale": "why_acquisition_happened",
          "market_impact": "how_this_affects_market_competition",
          "discovery_relevance": "why_this_is_relevant_to_search"
        }}]
        
        Focus on 8-10 most relevant acquisition-discovered providers.
        """
        
        try:
            response = await self.ask_ai(
                prompt=acquisition_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1200
            )
            
            acquisition_data = safe_json_parse(response, default=[])
            
            candidates = []
            for acquisition_info in acquisition_data:
                try:
                    # Calculate confidence based on transaction type and status
                    transaction_type = acquisition_info.get('acquisition_type', 'unknown')
                    current_status = acquisition_info.get('current_status', 'unknown')
                    
                    base_confidence = 0.5  # Lower base for acquisition-discovered
                    
                    # Boost confidence for recent acquisitions and funded companies
                    if transaction_type in ['funded', 'acquired'] and current_status == 'independent':
                        base_confidence += 0.2
                    elif transaction_type == 'spinoff':
                        base_confidence += 0.15
                    
                    # Consider transaction value if available
                    transaction_value = acquisition_info.get('transaction_value', '')
                    if any(indicator in transaction_value.lower() for indicator in ['million', 'billion', 'funded']):
                        base_confidence += 0.1
                    
                    final_confidence = min(0.75, base_confidence)
                    
                    candidate = ProviderCandidate(
                        name=acquisition_info.get('company_name', 'Unknown'),
                        website=acquisition_info.get('website', ''),
                        discovery_method='competitive_acquisition_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='acquisition_target',
                        ai_analysis={
                            'source': 'acquisition_analysis',
                            'acquisition_type': transaction_type,
                            'acquiring_company': acquisition_info.get('acquiring_company', ''),
                            'transaction_date': acquisition_info.get('transaction_date', ''),
                            'transaction_value': transaction_value,
                            'current_status': current_status,
                            'strategic_rationale': acquisition_info.get('strategic_rationale', ''),
                            'market_impact': acquisition_info.get('market_impact', ''),
                            'discovery_relevance': acquisition_info.get('discovery_relevance', '')
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create acquisition candidate: {e}")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Acquisition analysis failed: {e}")
            return []
    
    async def _analyze_supplier_networks(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        ecosystem: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Discover providers through supplier and vendor network analysis
        
        Args:
            target: Discovery target
            strategy: Search strategy
            ecosystem: Competitive ecosystem
            
        Returns:
            List of candidates discovered through supplier analysis
        """
        
        supplier_prompt = f"""
        Analyze supplier and vendor networks for {target.service_category} industry in {target.country}.
        
        Market Context:
        - Service Category: {target.service_category}
        - Country: {target.country}
        - Known Players: {[leader.get('company_name', '') for leader in ecosystem.get('market_leaders', [])[:5]]}
        
        Discover service providers through supply chain analysis:
        
        1. **Technology Suppliers**: Companies providing infrastructure or tools
        2. **Service Suppliers**: Companies providing outsourced services
        3. **Component Providers**: Companies providing specialized components
        4. **Platform Providers**: Companies providing platforms or frameworks
        5. **Integration Partners**: Companies specializing in implementations
        
        Identify suppliers that:
        - Could offer {target.service_category} services directly to end customers
        - Have the capability to compete with existing market leaders
        - Serve the same customer base but through supply relationships
        - Have grown beyond just being suppliers
        
        Return JSON with supplier-discovered providers:
        [{{
          "company_name": "supplier_company_name",
          "website": "estimated_website",
          "supplier_type": "technology/service/component/platform/integration",
          "client_companies": ["companies_they_supply_to"],
          "supplier_capabilities": ["what_they_provide"],
          "potential_market_role": "how_they_could_serve_end_customers",
          "competitive_potential": "likelihood_of_direct_competition",
          "market_transition": "evidence_of_moving_beyond_supplier_role"
        }}]
        
        Focus on 6-8 suppliers with highest potential for direct market participation.
        """
        
        try:
            response = await self.ask_ai(
                prompt=supplier_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1000
            )
            
            supplier_data = safe_json_parse(response, default=[])
            
            candidates = []
            for supplier_info in supplier_data:
                try:
                    # Calculate confidence based on competitive potential
                    competitive_potential = supplier_info.get('competitive_potential', '').lower()
                    market_transition = supplier_info.get('market_transition', '')
                    
                    base_confidence = 0.4  # Lower base for supplier-discovered
                    
                    # Boost confidence based on competitive indicators
                    if 'high' in competitive_potential or 'likely' in competitive_potential:
                        base_confidence += 0.2
                    elif 'medium' in competitive_potential or 'possible' in competitive_potential:
                        base_confidence += 0.1
                    
                    # Boost if there's evidence of market transition
                    if market_transition and len(market_transition) > 10:
                        base_confidence += 0.15
                    
                    final_confidence = min(0.7, base_confidence)
                    
                    candidate = ProviderCandidate(
                        name=supplier_info.get('company_name', 'Unknown'),
                        website=supplier_info.get('website', ''),
                        discovery_method='competitive_supplier_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='supplier_transition',
                        ai_analysis={
                            'source': 'supplier_analysis',
                            'supplier_type': supplier_info.get('supplier_type', ''),
                            'client_companies': supplier_info.get('client_companies', []),
                            'supplier_capabilities': supplier_info.get('supplier_capabilities', []),
                            'potential_market_role': supplier_info.get('potential_market_role', ''),
                            'competitive_potential': competitive_potential,
                            'market_transition': market_transition
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create supplier candidate: {e}")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Supplier analysis failed: {e}")
            return []
    
    async def _analyze_customer_references(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        ecosystem: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Discover providers through customer reference and case study analysis
        
        Args:
            target: Discovery target
            strategy: Search strategy
            ecosystem: Competitive ecosystem
            
        Returns:
            List of candidates discovered through customer references
        """
        
        customer_prompt = f"""
        Analyze customer references and case studies to discover {target.service_category} providers in {target.country}.
        
        Analysis Focus:
        - Service Category: {target.service_category}
        - Target Country: {target.country}
        - Known Market Leaders: {[leader.get('company_name', '') for leader in ecosystem.get('market_leaders', [])[:4]]}
        
        Discover providers through customer intelligence:
        
        1. **Case Study Analysis**: Providers mentioned in customer success stories
        2. **Reference Networks**: Companies referenced by existing customers
        3. **Implementation Partners**: Companies mentioned in deployment case studies
        4. **Customer Testimonials**: Providers praised by customers
        5. **Industry Awards**: Companies recognized by customers or industry
        
        Look for providers that:
        - Are mentioned in customer case studies or testimonials
        - Have won customer satisfaction awards
        - Are referenced in implementation success stories
        - Have strong customer advocacy
        
        Return JSON with customer-discovered providers:
        [{{
          "company_name": "provider_company_name",
          "website": "estimated_website",
          "discovery_source": "case_study/testimonial/award/reference",
          "customer_evidence": ["types_of_customer_validation"],
          "customer_segments": ["types_of_customers_served"],
          "success_indicators": ["awards", "recognition", "metrics"],
          "market_reputation": "customer_perception_indicators",
          "service_differentiation": "what_customers_highlight"
        }}]
        
        Focus on 6-8 providers with strongest customer validation evidence.
        """
        
        try:
            response = await self.ask_ai(
                prompt=customer_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1000
            )
            
            customer_data = safe_json_parse(response, default=[])
            
            candidates = []
            for customer_info in customer_data:
                try:
                    # Calculate confidence based on customer evidence strength
                    customer_evidence = customer_info.get('customer_evidence', [])
                    success_indicators = customer_info.get('success_indicators', [])
                    
                    base_confidence = 0.5  # Medium base for customer-discovered
                    
                    # Boost confidence based on evidence strength
                    evidence_boost = min(0.2, len(customer_evidence) * 0.05)
                    success_boost = min(0.15, len(success_indicators) * 0.03)
                    
                    final_confidence = min(0.8, base_confidence + evidence_boost + success_boost)
                    
                    candidate = ProviderCandidate(
                        name=customer_info.get('company_name', 'Unknown'),
                        website=customer_info.get('website', ''),
                        discovery_method='competitive_customer_reference_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='customer_validated',
                        ai_analysis={
                            'source': 'customer_reference_analysis',
                            'discovery_source': customer_info.get('discovery_source', ''),
                            'customer_evidence': customer_evidence,
                            'customer_segments': customer_info.get('customer_segments', []),
                            'success_indicators': success_indicators,
                            'market_reputation': customer_info.get('market_reputation', ''),
                            'service_differentiation': customer_info.get('service_differentiation', '')
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create customer reference candidate: {e}")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Customer reference analysis failed: {e}")
            return []
    
    async def _analyze_seo_competitors(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        ecosystem: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Discover providers through SEO competitor analysis
        
        Args:
            target: Discovery target
            strategy: Search strategy
            ecosystem: Competitive ecosystem
            
        Returns:
            List of candidates discovered through SEO analysis
        """
        
        seo_prompt = f"""
        Analyze SEO competitive landscape for {target.service_category} in {target.country} to discover providers.
        
        SEO Analysis Context:
        - Service Category: {target.service_category}
        - Country: {target.country}
        - Language: {target.language}
        - Known Competitors: {[leader.get('company_name', '') for leader in ecosystem.get('market_leaders', [])[:5]]}
        
        Discover providers through SEO competitive intelligence:
        
        1. **Search Visibility**: Companies ranking for relevant keywords
        2. **Content Competition**: Companies creating competing content
        3. **Advertising Competition**: Companies bidding on similar keywords
        4. **Local SEO**: Companies optimizing for local search
        5. **Industry Keywords**: Companies targeting industry-specific terms
        
        Identify providers that:
        - Rank highly for {target.service_category} keywords in {target.country}
        - Create substantial content about {target.service_category}
        - Invest in search advertising for relevant terms
        - Have strong local search presence
        - Target similar customer search intent
        
        Return JSON with SEO-discovered providers:
        [{{
          "company_name": "competitor_company_name",
          "website": "primary_website_domain",
          "seo_indicators": ["high_rankings", "content_volume", "ad_presence"],
          "target_keywords": ["keywords_they_compete_for"],
          "content_strategy": "their_content_approach",
          "local_presence": "local_seo_strength",
          "search_visibility": "estimated_visibility_level",
          "competitive_overlap": "similarity_to_known_competitors"
        }}]
        
        Focus on 8-10 providers with strongest SEO competitive indicators.
        """
        
        try:
            response = await self.ask_ai(
                prompt=seo_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1000
            )
            
            seo_data = safe_json_parse(response, default=[])
            
            candidates = []
            for seo_info in seo_data:
                try:
                    # Calculate confidence based on SEO indicators
                    seo_indicators = seo_info.get('seo_indicators', [])
                    search_visibility = seo_info.get('search_visibility', '').lower()
                    
                    base_confidence = 0.4  # Medium-low base for SEO discovery
                    
                    # Boost confidence based on SEO strength
                    indicators_boost = min(0.2, len(seo_indicators) * 0.07)
                    
                    # Boost for high visibility
                    if 'high' in search_visibility or 'strong' in search_visibility:
                        base_confidence += 0.15
                    elif 'medium' in search_visibility or 'moderate' in search_visibility:
                        base_confidence += 0.1
                    
                    final_confidence = min(0.7, base_confidence + indicators_boost)
                    
                    candidate = ProviderCandidate(
                        name=seo_info.get('company_name', 'Unknown'),
                        website=seo_info.get('website', ''),
                        discovery_method='competitive_seo_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='seo_competitor',
                        ai_analysis={
                            'source': 'seo_competitor_analysis',
                            'seo_indicators': seo_indicators,
                            'target_keywords': seo_info.get('target_keywords', []),
                            'content_strategy': seo_info.get('content_strategy', ''),
                            'local_presence': seo_info.get('local_presence', ''),
                            'search_visibility': search_visibility,
                            'competitive_overlap': seo_info.get('competitive_overlap', '')
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create SEO candidate: {e}")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"SEO competitor analysis failed: {e}")
            return []
    
    async def _analyze_competitor_networks(
        self, 
        market_leaders: List[Dict[str, Any]],
        target: DiscoveryTarget,
        ecosystem: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Analyze networks and relationships between competitors
        
        Args:
            market_leaders: List of identified market leaders
            target: Discovery target
            ecosystem: Competitive ecosystem
            
        Returns:
            List of candidates discovered through network analysis
        """
        
        network_prompt = f"""
        Perform network analysis of competitor relationships in {target.service_category} industry in {target.country}.
        
        Market Leaders for Network Analysis:
        {json.dumps([{
            'name': leader.get('company_name', ''),
            'relationships': leader.get('competitor_relationships', [])
        } for leader in market_leaders[:6]], indent=2)}
        
        Ecosystem Networks:
        {json.dumps(ecosystem.get('competitive_clusters', []), indent=2)}
        
        Discover providers through relationship network analysis:
        
        1. **Second-Degree Connections**: Companies connected to known competitors
        2. **Cluster Analysis**: Companies in same competitive clusters
        3. **Board Connections**: Companies with shared board members or advisors
        4. **Investment Networks**: Companies with shared investors
        5. **Alumni Networks**: Companies founded by former employees
        
        Identify network-connected providers:
        - Companies frequently mentioned alongside known competitors
        - Companies in same industry associations or groups
        - Companies with executive connections to market leaders
        - Companies serving overlapping customer bases
        - Companies participating in same industry events
        
        Return JSON with network-discovered providers:
        [{{
          "company_name": "network_connected_company",
          "website": "estimated_website",
          "network_connections": ["companies_connected_to"],
          "connection_types": ["board", "investment", "alumni", "customer", "industry"],
          "network_strength": "strong/medium/weak",
          "relationship_evidence": ["types_of_evidence_for_connections"],
          "competitive_relevance": "why_network_position_indicates_relevance"
        }}]
        
        Focus on 6-8 providers with strongest network connections to known competitors.
        """
        
        try:
            response = await self.ask_ai(
                prompt=network_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1000
            )
            
            network_data = safe_json_parse(response, default=[])
            
            candidates = []
            for network_info in network_data:
                try:
                    # Calculate confidence based on network strength
                    network_strength = network_info.get('network_strength', '').lower()
                    connection_types = network_info.get('connection_types', [])
                    network_connections = network_info.get('network_connections', [])
                    
                    base_confidence = 0.5
                    
                    # Boost confidence based on network strength
                    if network_strength == 'strong':
                        base_confidence += 0.2
                    elif network_strength == 'medium':
                        base_confidence += 0.1
                    
                    # Boost for multiple connection types
                    connection_boost = min(0.15, len(connection_types) * 0.05)
                    
                    # Boost for multiple connections
                    network_boost = min(0.1, len(network_connections) * 0.03)
                    
                    final_confidence = min(0.8, base_confidence + connection_boost + network_boost)
                    
                    candidate = ProviderCandidate(
                        name=network_info.get('company_name', 'Unknown'),
                        website=network_info.get('website', ''),
                        discovery_method='competitive_network_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='network_connected',
                        ai_analysis={
                            'source': 'competitor_network_analysis',
                            'network_connections': network_connections,
                            'connection_types': connection_types,
                            'network_strength': network_strength,
                            'relationship_evidence': network_info.get('relationship_evidence', []),
                            'competitive_relevance': network_info.get('competitive_relevance', '')
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create network candidate: {e}")
            
            self.analysis_stats['networks_analyzed'] += 1
            return candidates
            
        except Exception as e:
            self.logger.error(f"Network analysis failed: {e}")
            return []
    
    async def _validate_competitive_candidates(
        self, 
        candidates: List[ProviderCandidate],
        target: DiscoveryTarget,
        ecosystem: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Validate and enhance competitive analysis candidates
        
        Args:
            candidates: List of candidates from competitive analysis
            target: Discovery target
            ecosystem: Competitive ecosystem
            
        Returns:
            List of validated and enhanced candidates
        """
        if not candidates:
            return []
        
        self.logger.info(f"🔍 Validating {len(candidates)} competitive candidates")
        
        # Remove duplicates by name
        unique_candidates = {}
        for candidate in candidates:
            normalized_name = candidate.name.lower().strip()
            if normalized_name not in unique_candidates:
                unique_candidates[normalized_name] = candidate
            else:
                # Keep candidate with higher confidence
                if candidate.confidence_score > unique_candidates[normalized_name].confidence_score:
                    unique_candidates[normalized_name] = candidate
        
        validated_candidates = list(unique_candidates.values())
        
        # Sort by confidence score
        validated_candidates.sort(key=lambda c: c.confidence_score, reverse=True)
        
        # Apply competitive context validation
        for candidate in validated_candidates:
            try:
                # Add competitive context to AI analysis
                candidate.ai_analysis['competitive_validation'] = {
                    'ecosystem_position': self._determine_ecosystem_position(candidate, ecosystem),
                    'competitive_uniqueness': self._assess_competitive_uniqueness(candidate, validated_candidates),
                    'market_fit_score': self._calculate_market_fit(candidate, target),
                    'validation_timestamp': time.time()
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to add competitive validation for {candidate.name}: {e}")
        
        # Filter by minimum confidence threshold
        final_candidates = [
            c for c in validated_candidates 
            if c.confidence_score >= target.min_confidence_score
        ]
        
        self.logger.info(f"✅ Competitive validation complete: {len(final_candidates)} high-confidence candidates")
        return final_candidates
    
    def _determine_ecosystem_position(self, candidate: ProviderCandidate, ecosystem: Dict[str, Any]) -> str:
        """Determine candidate's position in competitive ecosystem"""
        
        # Check if candidate is in known clusters
        competitive_clusters = ecosystem.get('competitive_clusters', [])
        for cluster in competitive_clusters:
            if candidate.name.lower() in [c.lower() for c in cluster.get('companies', [])]:
                return f"cluster_member_{cluster.get('cluster_name', 'unknown')}"
        
        # Check market position
        market_position = candidate.market_position
        if market_position in ['leader', 'regional', 'global']:
            return 'market_leader'
        elif market_position in ['partner_ecosystem', 'supplier_transition']:
            return 'ecosystem_participant'
        else:
            return 'independent_player'
    
    def _assess_competitive_uniqueness(self, candidate: ProviderCandidate, all_candidates: List[ProviderCandidate]) -> float:
        """Assess how unique this candidate is compared to others"""
        
        # Count similar discovery methods
        similar_methods = sum(
            1 for c in all_candidates 
            if c != candidate and c.discovery_method == candidate.discovery_method
        )
        
        # Calculate uniqueness score (higher = more unique)
        total_candidates = len(all_candidates)
        if total_candidates <= 1:
            return 1.0
        
        uniqueness = 1.0 - (similar_methods / total_candidates)
        return max(0.1, uniqueness)  # Minimum 0.1 uniqueness
    
    def _calculate_market_fit(self, candidate: ProviderCandidate, target: DiscoveryTarget) -> float:
        """Calculate how well candidate fits target market"""
        
        fit_score = 0.5  # Base score
        
        # Check business category alignment
        if candidate.business_category.lower() == target.service_category.lower():
            fit_score += 0.3
        
        # Check market position indicators
        if candidate.market_position in ['leader', 'regional', 'customer_validated']:
            fit_score += 0.2
        
        return min(1.0, fit_score)
    
    async def _generate_default_competitive_strategy(self, target: DiscoveryTarget) -> List[SearchStrategy]:
        """Generate default competitive strategy when none provided"""
        
        default_strategy = SearchStrategy(
            method="competitor_analysis",
            priority=7,
            queries=[
                f"{target.service_category} competitors {target.country}",
                f"market leaders {target.service_category} {target.country}",
                f"top {target.service_category} companies {target.country}"
            ],
            platforms=["market_research", "industry_analysis", "competitive_intelligence"],
            expected_yield="12-20",
            ai_analysis_needed=True,
            follow_up_actions=["partnership_analysis", "acquisition_research"],
            metadata={"auto_generated": True, "comprehensive": True}
        )
        
        return [default_strategy]
    
    def _get_fallback_market_leaders(self, target: DiscoveryTarget) -> List[Dict[str, Any]]:
        """Get basic market leaders when AI analysis fails"""
        
        return [
            {
                "company_name": f"Leading {target.service_category} Provider",
                "tier": "global",
                "estimated_market_share": "unknown",
                "website": "",
                "headquarters": target.country,
                "geographic_coverage": [target.country],
                "key_strengths": ["market_presence"],
                "recent_developments": [],
                "competitor_relationships": [],
                "market_positioning": "established_player",
                "customer_segments": ["enterprise"],
                "confidence_level": 0.3
            }
        ]
    
    async def _load_competitive_cache(self):
        """Load competitive cache from persistent storage"""
        if not self.redis_client:
            return
        
        try:
            cache_data = await self.redis_client.get("competitor_analyzer_cache")
            if cache_data:
                self.competitive_cache = json.loads(cache_data)
                self.logger.info("📚 Loaded competitive cache from storage")
        except Exception as e:
            self.logger.warning(f"Failed to load competitive cache: {e}")
    
    async def _save_competitive_cache(self):
        """Save competitive cache to persistent storage"""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.setex(
                "competitor_analyzer_cache",
                86400,  # 24 hours
                json.dumps(self.competitive_cache)
            )
        except Exception as e:
            self.logger.warning(f"Failed to save competitive cache: {e}")
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get comprehensive competitive analysis statistics"""
        base_stats = self.get_metrics()
        
        analysis_stats = {
            **base_stats,
            "competitive_performance": self.analysis_stats,
            "intelligence_sources": list(self.intelligence_sources.keys()),
            "analysis_methods": [
                "market_leader_analysis",
                "partnership_discovery", 
                "acquisition_analysis",
                "supplier_network_analysis",
                "customer_reference_analysis",
                "seo_competitor_analysis"
            ]
        }
        
        return analysis_stats