# agents/discovery/regulatory_scanner.py
"""
Regulatory Scanner Agent

This agent specializes in discovering service providers through regulatory
databases, government registries, and official licensing authorities. It uses
AI to understand regulatory structures and extract provider information from
official sources.
"""

import asyncio
import logging
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup
import hashlib

from ..base import BaseAgent, AgentConfig, AgentError, safe_json_parse, retry_with_backoff, rate_limiter
from .models import DiscoveryTarget, ProviderCandidate, SearchStrategy
from services.ai_async_client import AIAsyncClient, TaskComplexity

logger = logging.getLogger(__name__)


class RegulatoryScanner(BaseAgent):
    """
    Specialized agent for regulatory database and official registry scanning
    
    Features:
    - AI-powered regulatory structure analysis
    - Multi-jurisdiction support for government databases
    - Intelligent data extraction from official sources
    - License and certification validation
    - Compliance status checking
    - Cross-reference with multiple regulatory bodies
    """
    
    def __init__(self, ai_client: AIAsyncClient, redis_client=None):
        config = AgentConfig(
            name="RegulatoryScanner",
            max_retries=3,
            rate_limit=10,  # Very conservative for official databases
            preferred_ai_provider="ollama",  # Use free provider for analysis
            task_complexity=TaskComplexity.MEDIUM,
            cache_ttl=7200,  # 2 hours cache for regulatory data
            debug=False
        )
        
        super().__init__(config, ai_client)
        
        # Redis client for caching regulatory data
        self.redis_client = redis_client
        
        # HTTP session for regulatory database access
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Regulatory database configurations by country/region
        self.regulatory_databases = {
            'USA': {
                'SEC': {
                    'name': 'Securities and Exchange Commission',
                    'base_url': 'https://www.sec.gov',
                    'search_endpoint': '/edgar/searchedgar',
                    'rate_limit': 2.0,
                    'data_format': 'html'
                },
                'FTC': {
                    'name': 'Federal Trade Commission',
                    'base_url': 'https://www.ftc.gov',
                    'search_endpoint': '/enforcement/cases-proceedings',
                    'rate_limit': 2.0,
                    'data_format': 'html'
                }
            },
            'Germany': {
                'BaFin': {
                    'name': 'Federal Financial Supervisory Authority',
                    'base_url': 'https://www.bafin.de',
                    'search_endpoint': '/EN/Supervision/CompanyDatabase',
                    'rate_limit': 3.0,
                    'data_format': 'html'
                }
            },
            'UK': {
                'Companies_House': {
                    'name': 'Companies House',
                    'base_url': 'https://beta.companieshouse.gov.uk',
                    'search_endpoint': '/search/companies',
                    'rate_limit': 1.5,
                    'data_format': 'json'
                },
                'FCA': {
                    'name': 'Financial Conduct Authority',
                    'base_url': 'https://www.fca.org.uk',
                    'search_endpoint': '/firms/systems-reporting',
                    'rate_limit': 2.0,
                    'data_format': 'html'
                }
            }
        }
        
        # Regulatory data cache
        self.regulatory_cache = {}
        
        # Performance tracking
        self.scan_stats = {
            'total_scans': 0,
            'successful_scans': 0,
            'cached_results': 0,
            'providers_found': 0,
            'database_performance': {}
        }
    
    async def _setup_agent(self) -> None:
        """Initialize HTTP session and regulatory database configurations"""
        try:
            # Setup HTTP session with conservative settings for official sites
            timeout = aiohttp.ClientTimeout(total=45)  # Longer timeout for government sites
            connector = aiohttp.TCPConnector(
                limit=5,  # Very conservative connection limit
                limit_per_host=2,
                ttl_dns_cache=600
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Academic Research Bot/1.0 (Business Directory Research)',
                    'Accept': 'text/html,application/json,*/*',
                    'Accept-Language': 'en-US,en;q=0.9'
                }
            )
            
            # Load regulatory cache
            await self._load_regulatory_cache()
            
            self.logger.info("Regulatory scanner initialized successfully")
            
        except Exception as e:
            raise AgentError(self.config.name, f"Failed to initialize regulatory scanner: {e}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup HTTP session and save cache"""
        try:
            # Save regulatory cache
            await self._save_regulatory_cache()
            
            # Close HTTP session
            if self.session and not self.session.closed:
                await self.session.close()
                await asyncio.sleep(0.2)  # Allow cleanup
            
            self.logger.info("Regulatory scanner cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup warning for regulatory scanner: {e}")
    
    async def scan_regulatory_databases(
        self, 
        target: DiscoveryTarget,
        strategies: List[SearchStrategy]
    ) -> List[ProviderCandidate]:
        """
        Main scanning method for regulatory databases and official registries
        
        Args:
            target: Discovery target specification
            strategies: Regulatory scanning strategies
            
        Returns:
            List of validated provider candidates from regulatory sources
        """
        self.logger.info(f"🏛️ Starting regulatory scan for {target.service_category} in {target.country}")
        
        start_time = time.time()
        all_candidates = []
        
        try:
            # Filter for regulatory strategies
            regulatory_strategies = [s for s in strategies if s.method == 'regulatory_bodies']
            
            if not regulatory_strategies:
                # Generate default regulatory strategy if none provided
                regulatory_strategies = await self._generate_default_regulatory_strategy(target)
            
            # Analyze regulatory landscape for target country
            regulatory_landscape = await self._analyze_regulatory_landscape(target)
            
            # Execute regulatory scans
            for strategy in regulatory_strategies:
                strategy_candidates = await self._execute_regulatory_strategy(
                    target, strategy, regulatory_landscape
                )
                all_candidates.extend(strategy_candidates)
                
                # Respectful rate limiting between strategies
                await asyncio.sleep(3.0)
            
            # Cross-validate candidates across multiple regulatory sources
            validated_candidates = await self._cross_validate_regulatory_candidates(
                all_candidates, target
            )
            
            scan_time = time.time() - start_time
            self.scan_stats['total_scans'] += 1
            if validated_candidates:
                self.scan_stats['successful_scans'] += 1
                self.scan_stats['providers_found'] += len(validated_candidates)
            
            self.logger.info(
                f"✅ Regulatory scan complete: {len(validated_candidates)} validated candidates "
                f"found in {scan_time:.1f}s"
            )
            
            return validated_candidates
            
        except Exception as e:
            self.logger.error(f"❌ Regulatory scan failed: {e}")
            raise AgentError(self.config.name, f"Regulatory scan failed: {e}")
    
    async def _analyze_regulatory_landscape(self, target: DiscoveryTarget) -> Dict[str, Any]:
        """
        Use AI to analyze the regulatory landscape for the target market
        
        Args:
            target: Discovery target with country and service category
            
        Returns:
            Regulatory landscape analysis with relevant authorities and requirements
        """
        
        landscape_prompt = f"""
        Analyze the regulatory landscape for {target.service_category} services in {target.country}.
        
        Target Market Context:
        - Country: {target.country}
        - Service Category: {target.service_category}
        - Language: {target.language}
        - Regulatory Bodies Mentioned: {', '.join(target.regulatory_bodies)}
        
        Provide comprehensive regulatory analysis:
        
        1. **Primary Regulatory Authorities**: Government agencies that oversee {target.service_category}
        2. **Licensing Requirements**: What licenses/certifications are typically required
        3. **Registration Databases**: Official databases where providers must register
        4. **Compliance Frameworks**: Key regulations and compliance standards
        5. **Industry Associations**: Professional bodies with member directories
        6. **Geographic Variations**: Regional or state-level regulatory differences in {target.country}
        
        Focus on identifying:
        - Official database URLs where providers are listed
        - Search parameters for finding registered providers
        - Data extraction patterns from regulatory websites
        - Cross-referencing opportunities between authorities
        
        Return JSON with actionable regulatory intelligence:
        {{
          "primary_authorities": [{{
            "name": "authority_name",
            "jurisdiction": "national/regional/local",
            "website": "official_website_url",
            "database_url": "provider_database_url",
            "search_method": "direct_search/api/scraping",
            "data_availability": "high/medium/low",
            "provider_categories": ["relevant_categories"],
            "update_frequency": "daily/weekly/monthly"
          }}],
          "licensing_requirements": [{{
            "license_type": "specific_license_name",
            "issuing_authority": "authority_name",
            "scope": "services_covered",
            "verification_method": "how_to_verify_status"
          }}],
          "search_strategies": [{{
            "database_name": "official_database",
            "search_parameters": ["category", "location", "status"],
            "expected_data_fields": ["company_name", "license_number", "status"],
            "access_method": "public/restricted/api"
          }}],
          "compliance_indicators": ["certification_names", "regulatory_codes"],
          "cross_reference_opportunities": ["database_pairs_for_validation"]
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=landscape_prompt,
                provider="ollama",  # Use free provider for analysis
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2000
            )
            
            landscape_data = safe_json_parse(response, default={})
            
            if not landscape_data:
                self.logger.warning("AI returned empty regulatory landscape, using fallback")
                return self._get_fallback_regulatory_landscape(target)
            
            self.logger.info(f"📊 Regulatory landscape analyzed: {len(landscape_data.get('primary_authorities', []))} authorities identified")
            return landscape_data
            
        except Exception as e:
            self.logger.error(f"Regulatory landscape analysis failed: {e}")
            return self._get_fallback_regulatory_landscape(target)
    
    async def _execute_regulatory_strategy(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        regulatory_landscape: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Execute a specific regulatory scanning strategy
        
        Args:
            target: Discovery target
            strategy: Regulatory scanning strategy
            regulatory_landscape: Regulatory analysis results
            
        Returns:
            List of candidates from regulatory sources
        """
        self.logger.info(f"🔍 Executing regulatory strategy: {strategy.method}")
        
        candidates = []
        authorities = regulatory_landscape.get('primary_authorities', [])
        
        # Scan each regulatory authority
        for authority in authorities:
            try:
                authority_candidates = await self._scan_regulatory_authority(
                    authority, target, strategy
                )
                candidates.extend(authority_candidates)
                
                # Respectful rate limiting between authorities
                await asyncio.sleep(authority.get('rate_limit', 3.0))
                
            except Exception as e:
                self.logger.error(f"Authority scan failed for {authority.get('name', 'unknown')}: {e}")
                continue
        
        # Also scan using AI knowledge of regulatory requirements
        ai_candidates = await self._scan_using_regulatory_knowledge(target, regulatory_landscape)
        candidates.extend(ai_candidates)
        
        self.logger.info(f"📋 Regulatory strategy executed: {len(candidates)} candidates found")
        return candidates
    
    @rate_limiter(max_calls=20, time_window=300)  # 20 calls per 5 minutes
    async def _scan_regulatory_authority(
        self, 
        authority: Dict[str, Any],
        target: DiscoveryTarget,
        strategy: SearchStrategy
    ) -> List[ProviderCandidate]:
        """
        Scan a specific regulatory authority's database
        
        Args:
            authority: Authority information from landscape analysis
            target: Discovery target
            strategy: Search strategy
            
        Returns:
            List of candidates from this authority
        """
        authority_name = authority.get('name', 'Unknown Authority')
        self.logger.info(f"🏛️ Scanning {authority_name}")
        
        # Check cache first
        cache_key = f"regulatory_{target.country}_{target.service_category}_{authority_name}"
        cached_results = await self._get_cached_regulatory_data(cache_key)
        if cached_results:
            self.scan_stats['cached_results'] += 1
            self.logger.debug(f"📚 Using cached results for {authority_name}")
            return cached_results
        
        try:
            database_url = authority.get('database_url', '')
            access_method = authority.get('access_method', 'scraping')
            
            if access_method == 'api' and authority.get('api_endpoint'):
                # Use API if available
                candidates = await self._scan_via_api(authority, target)
            elif database_url:
                # Use web scraping for HTML databases
                candidates = await self._scan_via_web_scraping(authority, target)
            else:
                # Use AI knowledge as fallback
                candidates = await self._scan_via_ai_knowledge(authority, target)
            
            # Cache successful results
            if candidates:
                await self._cache_regulatory_data(cache_key, candidates)
            
            # Update authority performance tracking
            if authority_name not in self.scan_stats['database_performance']:
                self.scan_stats['database_performance'][authority_name] = {
                    'total_scans': 0,
                    'successful_scans': 0,
                    'avg_results': 0.0
                }
            
            authority_stats = self.scan_stats['database_performance'][authority_name]
            authority_stats['total_scans'] += 1
            if candidates:
                authority_stats['successful_scans'] += 1
                prev_avg = authority_stats['avg_results']
                successful = authority_stats['successful_scans']
                authority_stats['avg_results'] = (prev_avg * (successful - 1) + len(candidates)) / successful
            
            self.logger.info(f"✅ {authority_name}: {len(candidates)} candidates found")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Authority scan failed for {authority_name}: {e}")
            return []
    
    async def _scan_via_api(
        self, 
        authority: Dict[str, Any],
        target: DiscoveryTarget
    ) -> List[ProviderCandidate]:
        """
        Scan regulatory database via official API
        
        Args:
            authority: Authority configuration with API details
            target: Discovery target
            
        Returns:
            List of candidates from API results
        """
        api_endpoint = authority.get('api_endpoint', '')
        api_key = authority.get('api_key', '')  # Would be configured via environment
        
        if not api_endpoint:
            self.logger.warning(f"No API endpoint for {authority.get('name')}")
            return []
        
        try:
            # Construct API request parameters
            params = {
                'category': target.service_category,
                'location': target.country,
                'status': 'active',
                'limit': 50
            }
            
            headers = {'Accept': 'application/json'}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
            async with self.session.get(api_endpoint, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._process_api_results(data, authority, target)
                else:
                    self.logger.warning(f"API request failed with status {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"API scan failed: {e}")
            return []
    
    async def _scan_via_web_scraping(
        self, 
        authority: Dict[str, Any],
        target: DiscoveryTarget
    ) -> List[ProviderCandidate]:
        """
        Scan regulatory database via web scraping (respectful)
        
        Args:
            authority: Authority configuration
            target: Discovery target
            
        Returns:
            List of candidates from scraped data
        """
        database_url = authority.get('database_url', '')
        if not database_url:
            self.logger.warning(f"No database URL for {authority.get('name')}")
            return []
        
        try:
            # Respectful scraping with proper delays
            await asyncio.sleep(2.0)
            
            async with self.session.get(database_url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Use AI to understand page structure and extract data
                    extraction_guidance = await self._get_extraction_guidance(
                        html_content[:5000], authority, target  # Limit content for AI analysis
                    )
                    
                    candidates = await self._extract_providers_from_html(
                        soup, extraction_guidance, authority, target
                    )
                    
                    return candidates
                else:
                    self.logger.warning(f"Web request failed with status {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Web scraping failed: {e}")
            return []
    
    async def _scan_via_ai_knowledge(
        self, 
        authority: Dict[str, Any],
        target: DiscoveryTarget
    ) -> List[ProviderCandidate]:
        """
        Use AI knowledge about regulatory requirements to identify known providers
        
        Args:
            authority: Authority information
            target: Discovery target
            
        Returns:
            List of candidates based on AI knowledge
        """
        knowledge_prompt = f"""
        Based on your knowledge of {authority.get('name', 'regulatory authority')} in {target.country},
        identify known {target.service_category} providers that would be registered or licensed.
        
        Authority Context:
        - Name: {authority.get('name')}
        - Jurisdiction: {authority.get('jurisdiction', 'unknown')}
        - Provider Categories: {authority.get('provider_categories', [])}
        
        Target Market: {target.service_category} in {target.country}
        
        Focus on providers that would typically be:
        1. **Licensed/Registered**: Officially registered with this authority
        2. **Compliant**: Meeting regulatory requirements
        3. **Established**: Well-known to regulators
        4. **Verified**: Can be cross-referenced with official sources
        
        Return JSON array of known providers:
        [{{
          "name": "company_name",
          "registration_number": "official_reg_number_if_known",
          "license_type": "specific_license_held",
          "registration_status": "active/inactive/pending",
          "business_address": "official_address_if_known",
          "services_authorized": ["specific_services_licensed"],
          "regulatory_standing": "good/warning/violation",
          "last_inspection_date": "date_if_known",
          "confidence_reasoning": "why_this_provider_is_known_to_regulator"
        }}]
        
        Limit to 8 most notable providers with strong regulatory connections.
        """
        
        try:
            response = await self.ask_ai(
                prompt=knowledge_prompt,
                provider="ollama",  # Use free provider for knowledge queries
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1200
            )
            
            knowledge_data = safe_json_parse(response, default=[])
            
            candidates = []
            for provider_data in knowledge_data:
                try:
                    # Calculate confidence based on regulatory standing
                    standing = provider_data.get('regulatory_standing', 'unknown')
                    status = provider_data.get('registration_status', 'unknown')
                    
                    base_confidence = 0.6  # Base for AI knowledge
                    
                    if standing == 'good' and status == 'active':
                        base_confidence += 0.2
                    elif standing == 'warning' or status == 'inactive':
                        base_confidence -= 0.1
                    
                    # Boost if registration number is provided
                    if provider_data.get('registration_number'):
                        base_confidence += 0.1
                    
                    final_confidence = min(0.85, max(0.3, base_confidence))
                    
                    candidate = ProviderCandidate(
                        name=provider_data.get('name', 'Unknown'),
                        website='',  # To be determined later
                        discovery_method=f'regulatory_ai_knowledge_{authority.get("name", "unknown")}',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='unknown',
                        ai_analysis={
                            'source': 'regulatory_knowledge',
                            'authority': authority.get('name'),
                            'registration_number': provider_data.get('registration_number', ''),
                            'license_type': provider_data.get('license_type', ''),
                            'registration_status': status,
                            'services_authorized': provider_data.get('services_authorized', []),
                            'regulatory_standing': standing,
                            'business_address': provider_data.get('business_address', ''),
                            'confidence_reasoning': provider_data.get('confidence_reasoning', '')
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create candidate from regulatory knowledge: {e}")
            
            self.logger.info(f"AI regulatory knowledge: {len(candidates)} candidates identified")
            return candidates
            
        except Exception as e:
            self.logger.error(f"AI knowledge scan failed: {e}")
            return []
    
    async def _get_extraction_guidance(
        self, 
        html_sample: str,
        authority: Dict[str, Any],
        target: DiscoveryTarget
    ) -> Dict[str, Any]:
        """
        Use AI to understand webpage structure and provide extraction guidance
        
        Args:
            html_sample: Sample of HTML content from regulatory page
            authority: Authority information
            target: Discovery target
            
        Returns:
            Extraction guidance with selectors and patterns
        """
        
        guidance_prompt = f"""
        Analyze this regulatory database webpage and provide extraction guidance:
        
        Authority: {authority.get('name')} - {target.country}
        Target: {target.service_category} providers
        
        HTML Sample:
        {html_sample}
        
        Analyze the page structure and provide extraction guidance:
        
        {{
          "data_structure": "table/list/cards/other",
          "provider_containers": "CSS_selector_for_each_provider",
          "extraction_patterns": {{
            "company_name": "CSS_selector_or_pattern",
            "registration_number": "CSS_selector_or_pattern",
            "license_type": "CSS_selector_or_pattern",
            "status": "CSS_selector_or_pattern",
            "contact_info": "CSS_selector_or_pattern"
          }},
          "pagination": {{
            "has_pagination": true/false,
            "next_page_selector": "CSS_selector_for_next_button",
            "page_count_pattern": "how_to_determine_total_pages"
          }},
          "filtering": {{
            "has_filters": true/false,
            "category_filter": "CSS_selector_for_service_category",
            "location_filter": "CSS_selector_for_location"
          }},
          "data_quality_indicators": ["signs_of_official_data"],
          "extraction_challenges": ["potential_issues_to_handle"]
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=guidance_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=800
            )
            
            guidance = safe_json_parse(response, default={})
            return guidance
            
        except Exception as e:
            self.logger.warning(f"Extraction guidance failed: {e}")
            return {}
    
    async def _extract_providers_from_html(
        self, 
        soup: BeautifulSoup,
        guidance: Dict[str, Any],
        authority: Dict[str, Any],
        target: DiscoveryTarget
    ) -> List[ProviderCandidate]:
        """
        Extract provider information from HTML using AI guidance
        
        Args:
            soup: Parsed HTML content
            guidance: AI extraction guidance
            authority: Authority information
            target: Discovery target
            
        Returns:
            List of extracted provider candidates
        """
        candidates = []
        
        try:
            provider_containers = guidance.get('extraction_patterns', {}).get('provider_containers', '')
            
            if provider_containers:
                # Use AI-guided selectors
                provider_elements = soup.select(provider_containers)
            else:
                # Fallback to common patterns
                provider_elements = soup.select('tr, .provider, .company, .listing')
            
            extraction_patterns = guidance.get('extraction_patterns', {})
            
            for i, element in enumerate(provider_elements[:20]):  # Limit to 20 to avoid overload
                try:
                    # Extract data using patterns
                    company_name = self._extract_text_by_pattern(
                        element, extraction_patterns.get('company_name', '')
                    )
                    
                    registration_number = self._extract_text_by_pattern(
                        element, extraction_patterns.get('registration_number', '')
                    )
                    
                    license_type = self._extract_text_by_pattern(
                        element, extraction_patterns.get('license_type', '')
                    )
                    
                    status = self._extract_text_by_pattern(
                        element, extraction_patterns.get('status', '')
                    )
                    
                    # Skip if no company name found
                    if not company_name or len(company_name) < 2:
                        continue
                    
                    # Calculate confidence based on data completeness
                    confidence_factors = {
                        'has_name': bool(company_name),
                        'has_registration': bool(registration_number),
                        'has_license_type': bool(license_type),
                        'has_status': bool(status),
                        'status_active': status.lower() in ['active', 'current', 'valid'] if status else False
                    }
                    
                    base_confidence = 0.7  # Higher for official sources
                    confidence_boost = sum(confidence_factors.values()) * 0.05
                    final_confidence = min(0.9, base_confidence + confidence_boost)
                    
                    candidate = ProviderCandidate(
                        name=company_name.strip(),
                        website='',  # Will be determined later
                        discovery_method=f'regulatory_scraping_{authority.get("name", "unknown")}',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='unknown',
                        ai_analysis={
                            'source': 'regulatory_scraping',
                            'authority': authority.get('name'),
                            'registration_number': registration_number,
                            'license_type': license_type,
                            'status': status,
                            'extraction_position': i + 1,
                            'data_completeness': confidence_factors
                        }
                    )
                    
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract provider {i}: {e}")
                    continue
            
            self.logger.info(f"HTML extraction: {len(candidates)} candidates extracted")
            return candidates
            
        except Exception as e:
            self.logger.error(f"HTML provider extraction failed: {e}")
            return []
    
    def _extract_text_by_pattern(self, element, pattern: str) -> str:
        """
        Extract text from HTML element using CSS selector pattern
        
        Args:
            element: BeautifulSoup element
            pattern: CSS selector pattern
            
        Returns:
            Extracted text or empty string
        """
        if not pattern:
            return ''
        
        try:
            # Try CSS selector first
            found_element = element.select_one(pattern)
            if found_element:
                return found_element.get_text(strip=True)
            
            # Try as attribute pattern
            if pattern.startswith('@'):
                attr_name = pattern[1:]
                return element.get(attr_name, '')
            
            # Try direct text search
            text = element.get_text()
            if pattern in text:
                # Extract text around the pattern
                start = text.find(pattern)
                return text[start:start+100].strip()
            
            return ''
            
        except Exception as e:
            self.logger.warning(f"Text extraction failed for pattern {pattern}: {e}")
            return ''
    
    async def _cross_validate_regulatory_candidates(
        self, 
        candidates: List[ProviderCandidate],
        target: DiscoveryTarget
    ) -> List[ProviderCandidate]:
        """
        Cross-validate candidates across multiple regulatory sources
        
        Args:
            candidates: List of candidates from various regulatory sources
            target: Discovery target
            
        Returns:
            List of cross-validated candidates
        """
        if not candidates:
            return []
        
        self.logger.info(f"🔍 Cross-validating {len(candidates)} regulatory candidates")
        
        # Group candidates by name for cross-validation
        name_groups = {}
        for candidate in candidates:
            normalized_name = candidate.name.lower().strip()
            if normalized_name not in name_groups:
                name_groups[normalized_name] = []
            name_groups[normalized_name].append(candidate)
        
        validated_candidates = []
        
        for normalized_name, candidate_group in name_groups.items():
            try:
                if len(candidate_group) == 1:
                    # Single source candidate
                    candidate = candidate_group[0]
                    if candidate.confidence_score >= 0.6:
                        validated_candidates.append(candidate)
                else:
                    # Multiple sources - create consolidated candidate
                    best_candidate = max(candidate_group, key=lambda c: c.confidence_score)
                    
                    # Boost confidence for multi-source validation
                    confidence_boost = min(0.2, (len(candidate_group) - 1) * 0.1)
                    best_candidate.confidence_score = min(0.95, best_candidate.confidence_score + confidence_boost)
                    
                    # Consolidate AI analysis
                    best_candidate.ai_analysis['cross_validation'] = {
                        'sources_count': len(candidate_group),
                        'sources': [c.discovery_method for c in candidate_group],
                        'confidence_boost': confidence_boost,
                        'validation_status': 'multi_source_confirmed'
                    }
                    
                    validated_candidates.append(best_candidate)
                    
            except Exception as e:
                self.logger.warning(f"Cross-validation failed for {normalized_name}: {e}")
                # Include best candidate anyway
                if candidate_group:
                    validated_candidates.append(max(candidate_group, key=lambda c: c.confidence_score))
        
        # Sort by confidence score
        validated_candidates.sort(key=lambda c: c.confidence_score, reverse=True)
        
        self.logger.info(f"✅ Cross-validation complete: {len(validated_candidates)} validated candidates")
        return validated_candidates
    
    async def _generate_default_regulatory_strategy(self, target: DiscoveryTarget) -> List[SearchStrategy]:
        """Generate default regulatory strategy when none provided"""
        
        default_strategy = SearchStrategy(
            method="regulatory_bodies",
            priority=8,
            queries=[
                f"registered {target.service_category} providers {target.country}",
                f"licensed {target.service_category} {target.country}",
                f"{target.service_category} business registry {target.country}"
            ],
            platforms=target.regulatory_bodies,
            expected_yield="10-25",
            ai_analysis_needed=True,
            follow_up_actions=["verify_licenses", "check_compliance_status"],
            metadata={"auto_generated": True, "fallback": True}
        )
        
        return [default_strategy]
    
    def _get_fallback_regulatory_landscape(self, target: DiscoveryTarget) -> Dict[str, Any]:
        """Get basic regulatory landscape when AI analysis fails"""
        
        return {
            "primary_authorities": [
                {
                    "name": f"{target.country} Business Registry",
                    "jurisdiction": "national",
                    "website": f"https://business-registry.{target.country.lower()}.gov",
                    "database_url": "",
                    "search_method": "web_search",
                    "data_availability": "medium",
                    "provider_categories": [target.service_category],
                    "update_frequency": "weekly"
                }
            ],
            "licensing_requirements": [
                {
                    "license_type": f"{target.service_category} License",
                    "issuing_authority": f"{target.country} Regulatory Authority",
                    "scope": target.service_category,
                    "verification_method": "online_database_check"
                }
            ],
            "search_strategies": [
                {
                    "database_name": "Business Registry",
                    "search_parameters": ["company_name", "service_category", "status"],
                    "expected_data_fields": ["name", "registration_number", "status"],
                    "access_method": "public"
                }
            ]
        }
    
    async def _get_cached_regulatory_data(self, cache_key: str) -> Optional[List[ProviderCandidate]]:
        """Retrieve cached regulatory data"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(f"regulatory:{cache_key}")
            if cached_data:
                candidates_data = json.loads(cached_data)
                return [ProviderCandidate(**data) for data in candidates_data]
        except Exception as e:
            self.logger.warning(f"Failed to get cached regulatory data: {e}")
        
        return None
    
    async def _cache_regulatory_data(self, cache_key: str, candidates: List[ProviderCandidate]):
        """Cache regulatory data for reuse"""
        if not self.redis_client:
            return
        
        try:
            candidates_data = [candidate.to_dict() for candidate in candidates]
            await self.redis_client.setex(
                f"regulatory:{cache_key}",
                self.config.cache_ttl,
                json.dumps(candidates_data)
            )
        except Exception as e:
            self.logger.warning(f"Failed to cache regulatory data: {e}")
    
    async def _load_regulatory_cache(self):
        """Load regulatory cache from persistent storage"""
        if not self.redis_client:
            return
        
        try:
            cache_data = await self.redis_client.get("regulatory_scanner_cache")
            if cache_data:
                self.regulatory_cache = json.loads(cache_data)
                self.logger.info("📚 Loaded regulatory cache from storage")
        except Exception as e:
            self.logger.warning(f"Failed to load regulatory cache: {e}")
    
    async def _save_regulatory_cache(self):
        """Save regulatory cache to persistent storage"""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.setex(
                "regulatory_scanner_cache",
                86400,  # 24 hours
                json.dumps(self.regulatory_cache)
            )
        except Exception as e:
            self.logger.warning(f"Failed to save regulatory cache: {e}")
    
    def get_scan_stats(self) -> Dict[str, Any]:
        """Get comprehensive regulatory scanning statistics"""
        base_stats = self.get_metrics()
        
        scan_stats = {
            **base_stats,
            "scan_performance": self.scan_stats,
            "supported_countries": list(self.regulatory_databases.keys()),
            "database_configurations": {
                country: len(databases) 
                for country, databases in self.regulatory_databases.items()
            }
        }
        
        return scan_stats