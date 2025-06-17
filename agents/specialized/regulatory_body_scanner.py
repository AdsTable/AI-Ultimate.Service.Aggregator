# agents/specialized/regulatory_body_scanner.py
"""
Advanced Regulatory Body Scanner for official provider discovery
Scans government databases, licensing authorities, and official registries
"""
import asyncio
import re
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from urllib.parse import urljoin, quote
from bs4 import BeautifulSoup

from .base_discovery_agent import BaseDiscoveryAgent, DiscoveryResult

class RegulatoryBodyScanner(BaseDiscoveryAgent):
    """
    Discovers service providers through official regulatory databases and government registries
    Focuses on licensed providers and official business registrations
    """
    
    def __init__(self, ai_client, config: Dict[str, Any]):
        super().__init__(ai_client, config)
        
        # Regulatory body configurations by country and service category
        self.regulatory_sources = {
            'Norway': {
                'electricity': {
                    'NVE': {
                        'name': 'Norwegian Water Resources and Energy Directorate',
                        'base_url': 'https://www.nve.no',
                        'search_method': 'website_search',
                        'provider_list_url': 'https://www.nve.no/stromkunde/strommarkedet/stromselskap/',
                        'rate_limit': 3.0
                    },
                    'Enercom': {
                        'name': 'Energy Company Registry',
                        'base_url': 'https://www.energimerking.no',
                        'search_method': 'directory_crawl',
                        'rate_limit': 2.0
                    }
                },
                'mobile': {
                    'Nkom': {
                        'name': 'Norwegian Communications Authority',
                        'base_url': 'https://www.nkom.no',
                        'search_method': 'website_search',
                        'provider_list_url': 'https://www.nkom.no/teknisk-div/nummerserier-og-adresser/mobilnett',
                        'rate_limit': 2.0
                    }
                },
                'general': {
                    'Brønnøysund': {
                        'name': 'Brønnøysund Register Centre',
                        'base_url': 'https://www.brreg.no',
                        'search_method': 'api_search',
                        'api_endpoint': 'https://data.brreg.no/enhetsregisteret/api/enheter',
                        'rate_limit': 1.0
                    }
                }
            },
            'Sweden': {
                'electricity': {
                    'Energimarknadsinspektionen': {
                        'name': 'Energy Markets Inspectorate',
                        'base_url': 'https://www.ei.se',
                        'search_method': 'website_search',
                        'rate_limit': 2.0
                    }
                },
                'mobile': {
                    'PTS': {
                        'name': 'Swedish Post and Telecom Authority',
                        'base_url': 'https://www.pts.se',
                        'search_method': 'website_search',
                        'rate_limit': 2.0
                    }
                }
            },
            'Denmark': {
                'electricity': {
                    'Energitilsynet': {
                        'name': 'Danish Energy Regulatory Authority',
                        'base_url': 'https://www.ens.dk',
                        'search_method': 'website_search',
                        'rate_limit': 2.0
                    }
                }
            }
        }
        
        # Common business activity codes for service categories
        self.activity_codes = {
            'electricity': ['35.14', '35.13', '35.11'],  # EU NACE codes
            'mobile': ['61.20', '61.10', '61.90'],
            'internet': ['61.10', '61.20', '61.30']
        }
    
    async def _initialize_agent_specific(self):
        """Initialize regulatory scanner specific resources"""
        logger.info("🏛️ Initializing Regulatory Body Scanner")
        
        # Test access to regulatory databases
        self.accessible_sources = await self._test_regulatory_access()
        logger.info(f"✅ Accessible regulatory sources: {len(self.accessible_sources)}")
    
    async def _cleanup_agent_specific(self):
        """Cleanup regulatory scanner specific resources"""
        logger.info("🔄 Cleaning up Regulatory Body Scanner")
    
    async def discover_providers(self, target_country: str, service_category: str) -> List[DiscoveryResult]:
        """
        Main discovery method for regulatory body scanning
        """
        logger.info(f"🏛️ Starting regulatory discovery for {service_category} in {target_country}")
        
        # Check cache first
        cache_key = f"regulatory_{target_country}_{service_category}"
        cached_results = await self._get_cached_results(cache_key, max_age_hours=48)  # Longer cache for official data
        if cached_results:
            return cached_results
        
        # Get regulatory sources for country and category
        country_sources = self.regulatory_sources.get(target_country, {})
        category_sources = country_sources.get(service_category, {})
        general_sources = country_sources.get('general', {})
        
        all_sources = {**category_sources, **general_sources}
        
        if not all_sources:
            logger.warning(f"No regulatory sources configured for {service_category} in {target_country}")
            return []
        
        # Discover from all available sources
        all_results = []
        
        for source_id, source_config in all_sources.items():
            try:
                source_results = await self._discover_from_regulatory_source(
                    source_id, source_config, target_country, service_category
                )
                all_results.extend(source_results)
                logger.info(f"🏛️ {source_id}: Found {len(source_results)} potential providers")
                
            except Exception as e:
                logger.error(f"❌ Discovery failed for {source_id}: {e}")
        
        # Validate and enhance results
        validated_results = await self._validate_regulatory_results(all_results, service_category)
        
        # Cache results
        await self._cache_results(cache_key, validated_results)
        
        logger.info(f"✅ Regulatory discovery complete: {len(validated_results)} validated providers")
        return validated_results
    
    async def _discover_from_regulatory_source(self, 
                                             source_id: str,
                                             source_config: Dict[str, Any],
                                             country: str,
                                             service_category: str) -> List[DiscoveryResult]:
        """Discover providers from a specific regulatory source"""
        
        search_method = source_config.get('search_method', 'website_search')
        
        await self._apply_rate_limiting(source_id, source_config.get('rate_limit', 2.0))
        
        if search_method == 'api_search':
            return await self._api_search(source_id, source_config, country, service_category)
        elif search_method == 'website_search':
            return await self._website_search(source_id, source_config, country, service_category)
        elif search_method == 'directory_crawl':
            return await self._directory_crawl(source_id, source_config, country, service_category)
        else:
            logger.warning(f"Unknown search method: {search_method}")
            return []
    
    async def _api_search(self, 
                         source_id: str,
                         source_config: Dict[str, Any],
                         country: str,
                         service_category: str) -> List[DiscoveryResult]:
        """Search using official API (e.g., Brønnøysund Register Centre)"""
        
        api_endpoint = source_config.get('api_endpoint')
        if not api_endpoint:
            return []
        
        results = []
        activity_codes = self.activity_codes.get(service_category, [])
        
        for activity_code in activity_codes:
            try:
                # Search by activity code
                search_params = {
                    'naeringskode': activity_code,
                    'size': 100,  # Max results per request
                    'aktiv': 'true'  # Only active companies
                }
                
                async with self.session.get(api_endpoint, params=search_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process API results
                        for company in data.get('_embedded', {}).get('enheter', []):
                            result = await self._process_api_company_data(
                                company, source_id, source_config, service_category
                            )
                            if result:
                                results.append(result)
                    else:
                        logger.warning(f"API search failed for {source_id}: HTTP {response.status}")
                
                # Rate limiting between activity code searches
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"API search error for {source_id}: {e}")
        
        return results
    
    async def _process_api_company_data(self, 
                                      company_data: Dict[str, Any],
                                      source_id: str,
                                      source_config: Dict[str, Any],
                                      service_category: str) -> Optional[DiscoveryResult]:
        """Process company data from API response"""
        
        company_name = company_data.get('navn', '')
        org_number = company_data.get('organisasjonsnummer', '')
        
        if not company_name:
            return None
        
        # Try to find company website
        website_url = await self._find_company_website(company_name, org_number)
        
        if not website_url:
            # Try to construct likely website
            website_url = await self._construct_website_url(company_name)
        
        # AI analysis to determine if this is actually a service provider
        business_analysis = await self._analyze_business_relevance(
            company_data, service_category
        )
        
        if business_analysis.get('is_service_provider', False):
            return self._create_discovery_result(
                provider_name=company_name,
                website_url=website_url or f"https://www.proff.no/selskap/{org_number}",  # Fallback to business directory
                confidence_score=business_analysis.get('confidence_score', 0.8),  # High confidence for official sources
                discovery_method='regulatory_api_search',
                source_platform=source_id,
                additional_data={
                    'business_details': {
                        'organization_number': org_number,
                        'registration_status': company_data.get('organisasjonsform', {}).get('beskrivelse', ''),
                        'business_address': company_data.get('forretningsadresse', {}),
                        'activity_codes': [nk.get('kode') for nk in company_data.get('naeringskoder', [])],
                        'employee_count': company_data.get('antallAnsatte')
                    },
                    'market_indicators': {
                        'official_registration': True,
                        'business_type': business_analysis.get('business_type', 'unknown'),
                        'market_focus': business_analysis.get('market_focus', 'unknown')
                    },
                    'raw_data': {
                        'api_response': company_data,
                        'source_config': source_config
                    }
                }
            )
        
        return None
    
    async def _website_search(self, 
                            source_id: str,
                            source_config: Dict[str, Any],
                            country: str,
                            service_category: str) -> List[DiscoveryResult]:
        """Search regulatory website for provider information"""
        
        base_url = source_config.get('base_url')
        provider_list_url = source_config.get('provider_list_url', base_url)
        
        if not provider_list_url:
            return []
        
        try:
            async with self.session.get(provider_list_url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    
                    # Extract provider information using AI
                    providers = await self._extract_providers_from_html(
                        html_content, provider_list_url, source_id, service_category
                    )
                    
                    return providers
                else:
                    logger.warning(f"Website search failed for {source_id}: HTTP {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Website search error for {source_id}: {e}")
            return []
    
    async def _extract_providers_from_html(self, 
                                         html_content: str,
                                         source_url: str,
                                         source_id: str,
                                         service_category: str) -> List[DiscoveryResult]:
        """Extract provider information from HTML using AI analysis"""
        
        # Parse HTML and extract text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove navigation, footer, and other non-content elements
        for element in soup.find_all(['nav', 'footer', 'header', 'aside', 'script', 'style']):
            element.decompose()
        
        # Extract clean text
        text_content = soup.get_text(separator='\n', strip=True)
        
        # AI analysis to extract provider information
        extraction_prompt = f"""
        Extract service provider information from this regulatory website content:
        
        Source URL: {source_url}
        Regulatory Body: {source_id}
        Service Category: {service_category}
        Content: {text_content[:8000]}  # Limit for AI processing
        
        Extract all mentioned service providers and return JSON array:
        [
            {{
                "provider_name": "Company Name",
                "website_url": "extracted_url_or_null",
                "license_info": "license_details_if_mentioned",
                "contact_info": {{"phone": "", "email": "", "address": ""}},
                "service_area": "geographic_coverage",
                "confidence_score": 0.0-1.0,
                "evidence": "text_snippet_supporting_identification"
            }}
        ]
        
        Focus on legitimate licensed providers mentioned in the official content.
        """
        
        try:
            response = await self.ai_client.ask(
                prompt=extraction_prompt,
                provider="ollama",
                task_complexity="complex"
            )
            
            ai_extractions = json.loads(response)
            
            # Convert to DiscoveryResult objects
            discovery_results = []
            
            for extraction in ai_extractions:
                if extraction.get('confidence_score', 0) > 0.7:
                    provider_name = extraction.get('provider_name', '')
                    website_url = extraction.get('website_url')
                    
                    # Validate and enhance website URL
                    if not website_url:
                        website_url = await self._construct_website_url(provider_name)
                    
                    result = self._create_discovery_result(
                        provider_name=provider_name,
                        website_url=website_url or f"https://www.google.com/search?q={quote(provider_name)}",
                        confidence_score=extraction.get('confidence_score', 0),
                        discovery_method='regulatory_website_search',
                        source_platform=source_id,
                        additional_data={
                            'business_details': {
                                'license_info': extraction.get('license_info'),
                                'service_area': extraction.get('service_area'),
                                'evidence': extraction.get('evidence')
                            },
                            'contact_info': extraction.get('contact_info', {}),
                            'market_indicators': {
                                'official_listing': True,
                                'regulatory_source': source_id
                            },
                            'raw_data': {
                                'source_url': source_url,
                                'extraction_data': extraction
                            }
                        }
                    )
                    discovery_results.append(result)
            
            return discovery_results
            
        except Exception as e:
            logger.error(f"AI extraction failed for {source_id}: {e}")
            return []
    
    async def _directory_crawl(self, 
                             source_id: str,
                             source_config: Dict[str, Any],
                             country: str,
                             service_category: str) -> List[DiscoveryResult]:
        """Crawl directory-style regulatory websites"""
        
        base_url = source_config.get('base_url')
        
        try:
            # Start with main page
            async with self.session.get(base_url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Find links to provider directories or lists
                    relevant_links = await self._find_relevant_directory_links(
                        soup, base_url, service_category
                    )
                    
                    all_results = []
                    
                    # Crawl each relevant link
                    for link_url in relevant_links[:5]:  # Limit to prevent excessive crawling
                        await self._apply_rate_limiting(source_id, 2.0)
                        
                        link_results = await self._crawl_directory_page(
                            link_url, source_id, service_category
                        )
                        all_results.extend(link_results)
                    
                    return all_results
                else:
                    logger.warning(f"Directory crawl failed for {source_id}: HTTP {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Directory crawl error for {source_id}: {e}")
            return []
    
    async def _find_relevant_directory_links(self, 
                                           soup: BeautifulSoup,
                                           base_url: str,
                                           service_category: str) -> List[str]:
        """Find relevant links in directory-style website"""
        
        # Keywords to look for in links
        category_keywords = {
            'electricity': ['strøm', 'kraftselskap', 'energi', 'electricity', 'power'],
            'mobile': ['mobil', 'telefon', 'tele', 'mobile', 'telecom'],
            'internet': ['internett', 'bredbånd', 'internet', 'broadband', 'fiber']
        }
        
        keywords = category_keywords.get(service_category, [])
        relevant_links = []
        
        # Find all links
        for link in soup.find_all('a', href=True):
            href = link['href']
            link_text = link.get_text(strip=True).lower()
            
            # Check if link text contains relevant keywords
            if any(keyword in link_text for keyword in keywords):
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    full_url = urljoin(base_url, href)
                elif href.startswith('http'):
                    full_url = href
                else:
                    continue
                
                relevant_links.append(full_url)
        
        return list(set(relevant_links))  # Remove duplicates
    
    async def _crawl_directory_page(self, 
                                  url: str,
                                  source_id: str,
                                  service_category: str) -> List[DiscoveryResult]:
        """Crawl a specific directory page for provider information"""
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    return await self._extract_providers_from_html(
                        html_content, url, source_id, service_category
                    )
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Directory page crawl error for {url}: {e}")
            return []
    
    async def _find_company_website(self, company_name: str, org_number: str) -> Optional[str]:
        """Find company website using various methods"""
        
        # Method 1: Search business directories
        search_urls = [
            f"https://www.proff.no/søk?q={quote(company_name)}",
            f"https://www.gulesider.no/søk/{quote(company_name)}"
        ]
        
        for search_url in search_urls:
            try:
                await asyncio.sleep(1.0)  # Rate limiting
                async with self.session.get(search_url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Look for website links
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            if href.startswith('http') and any(
                                domain in href for domain in ['.no', '.com', '.org']
                            ):
                                # Basic validation that this looks like a company website
                                if await self._validate_url(href):
                                    return href
            except:
                continue
        
        return None
    
    async def _construct_website_url(self, company_name: str) -> Optional[str]:
        """Construct likely website URL from company name"""
        
        # Clean company name
        name_cleaned = re.sub(r'[^a-zA-ZæøåÆØÅ0-9]', '', company_name.lower())
        name_cleaned = name_cleaned.replace('æ', 'ae').replace('ø', 'o').replace('å', 'aa')
        name_cleaned = name_cleaned.replace('Æ', 'AE').replace('Ø', 'O').replace('Å', 'AA')
        
        # Try various URL patterns
        potential_urls = [
            f"https://www.{name_cleaned}.no",
            f"https://www.{name_cleaned}.com",
            f"https://{name_cleaned}.no",
            f"https://{name_cleaned}.com"
        ]
        
        for url in potential_urls:
            if await self._validate_url(url):
                return url
        
        return None
    
    async def _analyze_business_relevance(self, 
                                        company_data: Dict[str, Any],
                                        service_category: str) -> Dict[str, Any]:
        """Analyze if company is relevant for the service category"""
        
        company_name = company_data.get('navn', '')
        activity_codes = [nk.get('kode') for nk in company_data.get('naeringskoder', [])]
        
        # AI analysis prompt
        analysis_prompt = f"""
        Determine if this Norwegian company is a {service_category} service provider:
        
        Company Name: {company_name}
        Activity Codes: {', '.join(activity_codes)}
        Business Form: {company_data.get('organisasjonsform', {}).get('beskrivelse', '')}
        Employee Count: {company_data.get('antallAnsatte')}
        
        Return JSON:
        {{
            "is_service_provider": boolean,
            "confidence_score": 0.0-1.0,
            "business_type": "B2C/B2B/both",
            "market_focus": "residential/commercial/both",
            "reasoning": "brief_explanation"
        }}
        """
        
        try:
            response = await self.ai_client.ask(
                prompt=analysis_prompt,
                provider="ollama",
                task_complexity="simple"
            )
            
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Business relevance analysis failed: {e}")
            return {
                "is_service_provider": False,
                "confidence_score": 0.0,
                "reasoning": f"Analysis failed: {str(e)}"
            }
    
    async def _validate_regulatory_results(self, 
                                         results: List[DiscoveryResult],
                                         service_category: str) -> List[DiscoveryResult]:
        """Validate and enhance regulatory discovery results"""
        
        if not results:
            return []
        
        # Remove duplicates by provider name
        unique_providers = {}
        for result in results:
            name_key = result.provider_name.lower().strip()
            if name_key not in unique_providers or result.confidence_score > unique_providers[name_key].confidence_score:
                unique_providers[name_key] = result
        
        validated_results = []
        
        for result in unique_providers.values():
            # Additional validation for official sources
            if result.confidence_score > 0.6:  # Lower threshold for official sources
                validated_results.append(result)
        
        # Sort by confidence and official status
        validated_results.sort(key=lambda x: (
            x.market_indicators.get('official_registration', False),
            x.confidence_score
        ), reverse=True)
        
        return validated_results[:15]  # Return top 15 results
    
    async def _test_regulatory_access(self) -> Dict[str, bool]:
        """Test access to regulatory sources"""
        
        accessible_sources = {}
        
        for country, categories in self.regulatory_sources.items():
            for category, sources in categories.items():
                for source_id, config in sources.items():
                    try:
                        base_url = config.get('base_url')
                        async with self.session.get(base_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            accessible_sources[f"{country}_{category}_{source_id}"] = response.status == 200
                    except:
                        accessible_sources[f"{country}_{category}_{source_id}"] = False
        
        return accessible_sources