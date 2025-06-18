# agents/extraction/content_extractor.py
"""
IntelligentContentExtractor Agent

This agent specializes in AI-powered website content analysis and extraction.
It combines web scraping with advanced AI analysis for comprehensive provider data extraction.
"""

import asyncio
import logging
import json
import time
import re
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, asdict
import aiohttp
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page

from ..base import BaseAgent, AgentConfig, AgentError, safe_json_parse, retry_with_backoff
from services.ai_async_client import AIAsyncClient, TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class WebsiteStructure:
    """Website structure analysis result"""
    title: str
    meta_description: str
    headings: List[str]
    navigation_links: List[Dict[str, str]]
    forms_count: int
    has_pricing_section: bool
    has_contact_section: bool
    structured_data: List[str]
    page_language: str
    estimated_page_type: str


@dataclass
class ExtractionResult:
    """Comprehensive extraction result with AI insights"""
    provider_name: str
    website_url: str
    business_info: Dict[str, Any]
    services: List[Dict[str, Any]]
    pricing_info: Dict[str, Any]
    contact_details: Dict[str, Any]
    competitive_analysis: Dict[str, Any]
    technical_details: Dict[str, Any]
    extraction_confidence: float
    raw_data: Dict[str, Any]
    extraction_timestamp: float


class IntelligentContentExtractor(BaseAgent):
    """
    Advanced AI-powered content extraction with multi-modal understanding
    
    Features:
    - Multi-stage extraction pipeline
    - AI-guided selector generation
    - Website structure analysis
    - Content validation and confidence scoring
    - Cross-platform compatibility
    - Intelligent retry mechanisms
    """
    
    def __init__(self, ai_client: AIAsyncClient, config: Optional[Dict[str, Any]] = None):
        agent_config = AgentConfig(
            name="IntelligentContentExtractor",
            max_retries=3,
            rate_limit=10,  # Respectful rate limiting for website extraction
            preferred_ai_provider="ollama",  # Use free provider for extraction guidance
            task_complexity=TaskComplexity.COMPLEX,
            cache_ttl=3600,  # 1 hour cache for extraction results
            debug=False
        )
        
        super().__init__(agent_config, ai_client)
        
        # HTTP session for web requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Playwright browser instance
        self.browser: Optional[Browser] = None
        
        # Extraction configuration
        self.extraction_config = {
            'timeout': 30,
            'wait_for_content': True,
            'javascript_enabled': True,
            'extract_images': False,
            'follow_redirects': True,
            'max_page_size': 10 * 1024 * 1024,  # 10MB limit
        }
        
        # Content extraction selectors (AI-enhanced)
        self.base_selectors = {
            'company_name': [
                'h1', '.company-name', '.brand-name', '.logo-text',
                '[class*="company"]', '[id*="company"]', 'title'
            ],
            'services': [
                '.services', '.products', '.offerings', '.solutions',
                '[class*="service"]', '[class*="product"]', 'main section'
            ],
            'pricing': [
                '.pricing', '.prices', '.plans', '.packages',
                '[class*="price"]', '[class*="plan"]', '[class*="package"]'
            ],
            'contact': [
                '.contact', '.contact-info', '.footer', '.header',
                '[class*="contact"]', '[id*="contact"]'
            ],
            'about': [
                '.about', '.about-us', '.company-info', '.overview',
                '[class*="about"]', '[href*="about"]'
            ]
        }
        
        # Performance tracking
        self.extraction_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'average_extraction_time': 0.0,
            'ai_analysis_calls': 0,
            'cache_hits': 0
        }
    
    async def _setup_agent(self) -> None:
        """Initialize HTTP session and browser"""
        try:
            # Setup HTTP session
            timeout = aiohttp.ClientTimeout(total=self.extraction_config['timeout'])
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            
            # Initialize Playwright browser
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-extensions',
                    '--disable-images',  # Speed optimization
                ]
            )
            
            self.logger.info("Intelligent content extractor initialized successfully")
            
        except Exception as e:
            raise AgentError(self.config.name, f"Failed to initialize content extractor: {e}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup HTTP session and browser"""
        try:
            # Close HTTP session
            if self.session and not self.session.closed:
                await self.session.close()
                await asyncio.sleep(0.1)
            
            # Close browser
            if self.browser:
                await self.browser.close()
            
            self.logger.info("Content extractor cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup warning for content extractor: {e}")
    
    async def extract_comprehensive_data(
        self, 
        website_url: str, 
        service_category: str,
        provider_name: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract comprehensive data using multi-stage AI analysis
        
        Args:
            website_url: URL of the website to extract from
            service_category: Category of service for context
            provider_name: Known provider name (optional)
            
        Returns:
            Comprehensive extraction result
        """
        self.logger.info(f"🔍 Starting comprehensive extraction for {website_url}")
        
        start_time = time.time()
        self.extraction_stats['total_extractions'] += 1
        
        try:
            # Stage 1: Website structure analysis
            self.logger.debug("Stage 1: Analyzing website structure")
            structure_analysis = await self._analyze_website_structure(website_url)
            
            # Stage 2: Content extraction with AI guidance
            self.logger.debug("Stage 2: Extracting content with AI guidance")
            content_data = await self._extract_guided_content(
                website_url, structure_analysis, service_category
            )
            
            # Stage 3: AI-powered data synthesis
            self.logger.debug("Stage 3: Synthesizing extracted data")
            synthesized_data = await self._synthesize_extracted_data(
                content_data, service_category, provider_name
            )
            
            # Stage 4: Competitive positioning analysis
            self.logger.debug("Stage 4: Analyzing competitive positioning")
            competitive_analysis = await self._analyze_competitive_positioning(
                synthesized_data, service_category
            )
            
            # Stage 5: Technical details extraction
            self.logger.debug("Stage 5: Extracting technical details")
            technical_details = await self._extract_technical_details(website_url, content_data)
            
            extraction_time = time.time() - start_time
            self._update_extraction_stats(extraction_time, success=True)
            
            # Create final result
            result = ExtractionResult(
                provider_name=synthesized_data.get('provider_name', provider_name or 'Unknown'),
                website_url=website_url,
                business_info=synthesized_data.get('business_info', {}),
                services=synthesized_data.get('services', []),
                pricing_info=synthesized_data.get('pricing', {}),
                contact_details=synthesized_data.get('contact', {}),
                competitive_analysis=competitive_analysis,
                technical_details=technical_details,
                extraction_confidence=synthesized_data.get('confidence', 0.0),
                raw_data={
                    'structure_analysis': asdict(structure_analysis) if structure_analysis else {},
                    'content_data': content_data,
                    'extraction_time': extraction_time
                },
                extraction_timestamp=time.time()
            )
            
            self.logger.info(
                f"✅ Comprehensive extraction completed for {website_url} "
                f"(confidence: {result.extraction_confidence:.2f}, time: {extraction_time:.1f}s)"
            )
            
            return result
            
        except Exception as e:
            extraction_time = time.time() - start_time
            self._update_extraction_stats(extraction_time, success=False)
            self.logger.error(f"❌ Extraction failed for {website_url}: {e}")
            raise AgentError(self.config.name, f"Content extraction failed: {e}")
    
    async def _analyze_website_structure(self, url: str) -> Optional[WebsiteStructure]:
        """AI-powered website structure analysis using Playwright"""
        
        try:
            context = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            
            page = await context.new_page()
            
            # Navigate and wait for content
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Extract comprehensive structure information
            structure_info = await page.evaluate("""
            () => {
                // Helper function to extract text content safely
                const getTextContent = (selector) => {
                    const element = document.querySelector(selector);
                    return element ? element.textContent.trim() : '';
                };
                
                // Helper function to get all matching elements
                const getAllTextContent = (selector) => {
                    const elements = document.querySelectorAll(selector);
                    return Array.from(elements).map(el => el.textContent.trim()).filter(text => text.length > 0);
                };
                
                return {
                    title: document.title || '',
                    meta_description: getTextContent('meta[name="description"]') || '',
                    headings: getAllTextContent('h1, h2, h3, h4'),
                    navigation_links: Array.from(document.querySelectorAll('nav a, .nav a, .menu a, header a')).map(a => ({
                        text: a.textContent.trim(),
                        href: a.href || '',
                        internal: a.href ? a.href.includes(window.location.hostname) : false
                    })).filter(link => link.text.length > 0),
                    forms_count: document.querySelectorAll('form').length,
                    has_pricing_section: !!(
                        document.querySelector('[class*="price"], [id*="price"], [class*="plan"], [id*="plan"], [class*="package"]') ||
                        document.querySelector('a[href*="pricing"], a[href*="plans"]')
                    ),
                    has_contact_section: !!(
                        document.querySelector('[class*="contact"], [id*="contact"], [href*="contact"]') ||
                        document.querySelector('a[href*="contact"], a[href*="mailto:"]')
                    ),
                    structured_data: Array.from(document.querySelectorAll('script[type="application/ld+json"]'))
                        .map(s => {
                            try { return JSON.parse(s.textContent); } catch { return null; }
                        }).filter(data => data !== null),
                    page_language: document.documentElement.lang || 'en',
                    page_text_length: document.body.innerText.length,
                    has_footer: !!document.querySelector('footer'),
                    has_header: !!document.querySelector('header'),
                    main_content_selectors: ['main', '.main', '#main', '.content', '#content', '.container'].filter(sel => 
                        document.querySelector(sel)
                    )
                }
            }
            """)
            
            await context.close()
            
            # Use AI to analyze and classify the structure
            structure_prompt = f"""
            Analyze this website structure and determine the optimal extraction strategy:
            
            URL: {url}
            Structure Data: {json.dumps(structure_info, indent=2)}
            
            Based on the structure, determine:
            1. **Page Type**: homepage/service_page/about_page/contact_page/pricing_page/other
            2. **Content Organization**: well_structured/moderately_structured/poorly_structured
            3. **Business Focus**: b2b/b2c/mixed/unclear
            4. **Information Completeness**: comprehensive/moderate/basic/minimal
            
            Return JSON with analysis and extraction recommendations:
            {{
              "page_classification": {{
                "primary_type": "page_type",
                "secondary_types": ["other_applicable_types"],
                "confidence": 0.0-1.0
              }},
              "structure_quality": {{
                "organization": "well_structured/moderately_structured/poorly_structured",
                "navigation_clarity": "clear/moderate/poor",
                "content_hierarchy": "logical/somewhat_logical/confusing"
              }},
              "extraction_strategy": {{
                "recommended_approach": "playwright_full/playwright_light/requests_only",
                "priority_selectors": [{{
                  "content_type": "company_name/services/pricing/contact",
                  "selectors": ["css_selector1", "css_selector2"],
                  "extraction_method": "text/attribute/combined"
                }}],
                "challenges": ["challenge1", "challenge2"],
                "estimated_extraction_success": 0.0-1.0
              }}
            }}
            """
            
            self.extraction_stats['ai_analysis_calls'] += 1
            response = await self.ask_ai(
                prompt=structure_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1000
            )
            
            ai_analysis = safe_json_parse(response, default={})
            
            # Determine estimated page type from AI analysis
            page_type = "unknown"
            if ai_analysis.get('page_classification', {}).get('primary_type'):
                page_type = ai_analysis['page_classification']['primary_type']
            
            return WebsiteStructure(
                title=structure_info.get('title', ''),
                meta_description=structure_info.get('meta_description', ''),
                headings=structure_info.get('headings', []),
                navigation_links=structure_info.get('navigation_links', []),
                forms_count=structure_info.get('forms_count', 0),
                has_pricing_section=structure_info.get('has_pricing_section', False),
                has_contact_section=structure_info.get('has_contact_section', False),
                structured_data=structure_info.get('structured_data', []),
                page_language=structure_info.get('page_language', 'en'),
                estimated_page_type=page_type
            )
            
        except Exception as e:
            self.logger.error(f"Structure analysis failed for {url}: {e}")
            return None
    
    async def _extract_guided_content(
        self, 
        url: str, 
        structure: Optional[WebsiteStructure], 
        service_category: str
    ) -> Dict[str, Any]:
        """Extract content using AI guidance and multiple methods"""
        
        extracted_content = {}
        
        try:
            # Method 1: Playwright extraction (comprehensive)
            playwright_content = await self._extract_with_playwright(url, structure, service_category)
            extracted_content.update(playwright_content)
            
            # Method 2: HTTP requests (fallback/supplement)
            if not extracted_content or len(str(extracted_content)) < 1000:
                http_content = await self._extract_with_requests(url)
                extracted_content.update(http_content)
            
            # Method 3: AI-guided selector refinement
            if structure and extracted_content:
                refined_content = await self._refine_extraction_with_ai(
                    extracted_content, structure, service_category
                )
                extracted_content.update(refined_content)
            
            return extracted_content
            
        except Exception as e:
            self.logger.error(f"Guided extraction failed for {url}: {e}")
            return {}
    
    async def _extract_with_playwright(
        self, 
        url: str, 
        structure: Optional[WebsiteStructure], 
        service_category: str
    ) -> Dict[str, Any]:
        """Extract content using Playwright with intelligent selectors"""
        
        try:
            context = await self.browser.new_context()
            page = await context.new_page()
            
            # Navigate to the page
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for dynamic content if needed
            if structure and 'javascript' in structure.estimated_page_type.lower():
                await asyncio.sleep(2)
            
            # Extract content using multiple selector strategies
            content_extraction = await page.evaluate(f"""
            (serviceCategory) => {{
                const extractedData = {{}};
                
                // Company name extraction
                const companySelectors = [
                    'h1', '.company-name', '.brand-name', '.logo-text', 
                    'title', '[class*="company"]', '[class*="brand"]'
                ];
                for (const selector of companySelectors) {{
                    const element = document.querySelector(selector);
                    if (element && element.textContent.trim() && !extractedData.company_name) {{
                        extractedData.company_name = element.textContent.trim();
                        break;
                    }}
                }}
                
                // Services extraction
                const serviceSelectors = [
                    '.services', '.products', '.offerings', '.solutions',
                    '[class*="service"]', '[class*="product"]', 'main section'
                ];
                extractedData.services = [];
                for (const selector of serviceSelectors) {{
                    const elements = document.querySelectorAll(selector);
                    elements.forEach(el => {{
                        const text = el.textContent.trim();
                        if (text.length > 20 && text.length < 1000) {{
                            extractedData.services.push(text);
                        }}
                    }});
                }}
                
                // Contact information extraction
                extractedData.contact = {{}};
                
                // Email extraction
                const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{{2,}}/g;
                const emailMatches = document.body.innerText.match(emailRegex);
                if (emailMatches) {{
                    extractedData.contact.emails = [...new Set(emailMatches)].slice(0, 3);
                }}
                
                // Phone extraction
                const phoneRegex = /[\\+]?[1-9]?[\\-\\s]?\\(?[0-9]{{3}}\\)?[\\-\\s]?[0-9]{{3,4}}[\\-\\s]?[0-9]{{4,6}}/g;
                const phoneMatches = document.body.innerText.match(phoneRegex);
                if (phoneMatches) {{
                    extractedData.contact.phones = [...new Set(phoneMatches)].slice(0, 3);
                }}
                
                // Address extraction (basic)
                const addressSelectors = ['.address', '.location', '[class*="address"]', '[class*="location"]'];
                for (const selector of addressSelectors) {{
                    const element = document.querySelector(selector);
                    if (element) {{
                        extractedData.contact.address = element.textContent.trim();
                        break;
                    }}
                }}
                
                // Pricing information
                const pricingSelectors = [
                    '.pricing', '.prices', '.plans', '.packages',
                    '[class*="price"]', '[class*="plan"]'
                ];
                extractedData.pricing = [];
                for (const selector of pricingSelectors) {{
                    const elements = document.querySelectorAll(selector);
                    elements.forEach(el => {{
                        const text = el.textContent.trim();
                        if (text.includes('$') || text.includes('€') || text.includes('£') || 
                            text.toLowerCase().includes('price') || text.toLowerCase().includes('plan')) {{
                            extractedData.pricing.push(text);
                        }}
                    }});
                }}
                
                // About/description extraction
                const aboutSelectors = [
                    '.about', '.description', '.overview', '.intro',
                    '[class*="about"]', '[class*="description"]'
                ];
                for (const selector of aboutSelectors) {{
                    const element = document.querySelector(selector);
                    if (element && element.textContent.trim().length > 50) {{
                        extractedData.about = element.textContent.trim();
                        break;
                    }}
                }}
                
                // Page metadata
                extractedData.metadata = {{
                    title: document.title || '',
                    description: document.querySelector('meta[name="description"]')?.content || '',
                    keywords: document.querySelector('meta[name="keywords"]')?.content || '',
                    page_text_length: document.body.innerText.length
                }};
                
                return extractedData;
            }}
            """, service_category)
            
            await context.close()
            return content_extraction
            
        except Exception as e:
            self.logger.error(f"Playwright extraction failed for {url}: {e}")
            return {}
    
    async def _extract_with_requests(self, url: str) -> Dict[str, Any]:
        """Fallback extraction using HTTP requests and BeautifulSoup"""
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return {}
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Basic content extraction
                extracted = {
                    'fallback_content': {
                        'title': soup.title.string if soup.title else '',
                        'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])],
                        'paragraphs': [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 20],
                        'links': [{'text': a.get_text().strip(), 'href': a.get('href', '')} 
                                for a in soup.find_all('a') if a.get_text().strip()],
                        'text_content': soup.get_text(separator=' ', strip=True)[:5000]  # First 5000 chars
                    }
                }
                
                return extracted
                
        except Exception as e:
            self.logger.error(f"HTTP extraction failed for {url}: {e}")
            return {}
    
    async def _synthesize_extracted_data(
        self, 
        content_data: Dict[str, Any], 
        service_category: str,
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """AI-powered data synthesis and structuring"""
        
        synthesis_prompt = f"""
        Synthesize and structure this extracted website content for a {service_category} provider:
        
        Extracted Content: {json.dumps(content_data, indent=2)[:8000]}  # Limit for token constraints
        Known Provider Name: {provider_name or 'Unknown'}
        
        Analyze and extract structured information:
        
        1. **Provider Information**:
           - Company name (clean, official name)
           - Business description and value proposition
           - Years in operation or founding year
           - Company size indicators
           - Certifications, licenses, awards
           - Business registration information
        
        2. **Services & Products**:
           - Detailed service offerings with descriptions
           - Service categories and specializations
           - Pricing structure and packages
           - Special offers or promotions
           - Service delivery methods
           - Geographic coverage
        
        3. **Contact & Location**:
           - Phone numbers (cleaned and formatted)
           - Email addresses (verified business emails)
           - Physical addresses with full details
           - Service areas and coverage
           - Business hours
           - Customer service availability
        
        4. **Business Details**:
           - Target customer segments
           - Industry experience and expertise
           - Technology stack or tools used
           - Partnership and affiliations
           - Quality certifications
        
        Return structured JSON with confidence scores:
        {{
          "provider_name": "cleaned_company_name",
          "business_info": {{
            "description": "comprehensive_business_description",
            "founding_year": "year_or_null",
            "company_size": "estimated_size_range",
            "certifications": ["cert1", "cert2"],
            "specializations": ["area1", "area2"],
            "target_customers": ["segment1", "segment2"],
            "value_proposition": "unique_selling_points"
          }},
          "services": [{{
            "name": "service_name",
            "description": "detailed_description",
            "category": "service_category",
            "pricing_model": "fixed/hourly/subscription/custom",
            "delivery_method": "onsite/remote/hybrid",
            "specializations": ["sub_specialization"]
          }}],
          "pricing": {{
            "structure": "transparent/negotiable/quote_based/unclear",
            "packages": [{{
              "name": "package_name",
              "price": "price_info",
              "features": ["feature1", "feature2"],
              "target_segment": "who_its_for"
            }}],
            "pricing_transparency": "high/medium/low"
          }},
          "contact": {{
            "primary_phone": "formatted_phone_number",
            "primary_email": "business_email",
            "address": {{
              "street": "street_address",
              "city": "city",
              "state": "state_or_region",
              "postal_code": "postal_code",
              "country": "country"
            }},
            "business_hours": "operating_hours",
            "service_areas": ["area1", "area2"],
            "contact_methods": ["phone", "email", "form", "chat"]
          }},
          "technical_details": {{
            "website_quality": "professional/good/basic/poor",
            "online_booking": "available/not_available",
            "customer_portal": "available/not_available",
            "mobile_friendly": "yes/no/unknown",
            "technology_focus": ["tech_area1", "tech_area2"]
          }},
          "confidence": {{
            "overall": 0.0-1.0,
            "provider_name": 0.0-1.0,
            "services": 0.0-1.0,
            "contact": 0.0-1.0,
            "pricing": 0.0-1.0
          }},
          "data_quality": {{
            "completeness": "comprehensive/moderate/basic/poor",
            "accuracy_indicators": ["verification_method1", "verification_method2"],
            "missing_information": ["missing_field1", "missing_field2"],
            "reliability_score": 0.0-1.0
          }}
        }}
        
        Focus on accuracy and provide confidence scores for extracted information.
        """
        
        try:
            self.extraction_stats['ai_analysis_calls'] += 1
            response = await self.ask_ai(
                prompt=synthesis_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2500
            )
            
            synthesized_data = safe_json_parse(response, default={})
            
            # Add synthesis metadata
            synthesized_data['synthesis_metadata'] = {
                'synthesized_at': time.time(),
                'ai_provider': 'ollama',
                'content_source_size': len(str(content_data)),
                'synthesis_method': 'ai_guided_extraction'
            }
            
            return synthesized_data
            
        except Exception as e:
            self.logger.error(f"Data synthesis failed: {e}")
            return {
                'confidence': {'overall': 0.1},
                'synthesis_error': str(e)
            }
    
    def _update_extraction_stats(self, extraction_time: float, success: bool):
        """Update extraction performance statistics"""
        if success:
            self.extraction_stats['successful_extractions'] += 1
        else:
            self.extraction_stats['failed_extractions'] += 1
        
        # Update average extraction time
        total_successful = self.extraction_stats['successful_extractions']
        if total_successful > 0:
            current_avg = self.extraction_stats['average_extraction_time']
            self.extraction_stats['average_extraction_time'] = (
                (current_avg * (total_successful - 1) + extraction_time) / total_successful
            )
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction performance statistics"""
        return {
            'total_extractions': self.extraction_stats['total_extractions'],
            'successful_extractions': self.extraction_stats['successful_extractions'],
            'failed_extractions': self.extraction_stats['failed_extractions'],
            'success_rate': (
                self.extraction_stats['successful_extractions'] / 
                max(1, self.extraction_stats['total_extractions'])
            ) * 100,
            'average_extraction_time': self.extraction_stats['average_extraction_time'],
            'ai_analysis_calls': self.extraction_stats['ai_analysis_calls'],
            'cache_hits': self.extraction_stats['cache_hits']
        }