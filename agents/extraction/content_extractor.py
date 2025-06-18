# agents/extraction/content_extractor.py
"""
IntelligentContentExtractor Agent

This agent orchestrates the complete content extraction pipeline,
combining website analysis, AI-guided extraction, and data synthesis
for comprehensive provider information gathering.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import aiohttp

from ..base import BaseAgent, AgentConfig, AgentError, safe_json_parse, retry_with_backoff
from .website_analyzer import WebsiteAnalyzer
from .data_synthesizer import DataSynthesizer
from services.ai_async_client import AIAsyncClient, TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class ExtractionTarget:
    """Target specification for content extraction"""
    website_url: str
    service_category: str
    provider_name: Optional[str] = None
    extraction_depth: str = "comprehensive"  # basic, standard, comprehensive
    target_language: str = "en"
    expected_content_types: List[str] = None
    
    def __post_init__(self):
        if self.expected_content_types is None:
            self.expected_content_types = [
                "company_info", "services", "pricing", "contact", "about"
            ]


@dataclass
class ExtractionResult:
    """Comprehensive extraction result with metadata"""
    provider_name: str
    website_url: str
    extraction_target: ExtractionTarget
    
    # Core extracted data
    business_info: Dict[str, Any]
    services: List[Dict[str, Any]]
    pricing_info: Dict[str, Any]
    contact_details: Dict[str, Any]
    
    # Analysis results
    competitive_analysis: Dict[str, Any]
    technical_details: Dict[str, Any]
    website_quality_score: float
    
    # Metadata
    extraction_confidence: float
    extraction_timestamp: float
    extraction_duration: float
    ai_providers_used: List[str]
    raw_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def to_standardized_product(self) -> Dict[str, Any]:
        """Convert to StandardizedProduct format for database storage"""
        return {
            'provider_name': self.provider_name,
            'website_url': self.website_url,
            'service_category': self.extraction_target.service_category,
            'business_description': self.business_info.get('description', ''),
            'services_offered': [s.get('name', '') for s in self.services],
            'pricing_model': self.pricing_info.get('structure', 'unknown'),
            'contact_email': self.contact_details.get('primary_email', ''),
            'contact_phone': self.contact_details.get('primary_phone', ''),
            'confidence_score': self.extraction_confidence,
            'last_updated': self.extraction_timestamp,
            'metadata': {
                'extraction_duration': self.extraction_duration,
                'website_quality': self.website_quality_score,
                'ai_providers_used': self.ai_providers_used
            }
        }


class IntelligentContentExtractor(BaseAgent):
    """
    Orchestrates intelligent content extraction using multi-stage AI analysis
    
    Features:
    - Multi-stage extraction pipeline
    - AI-guided website analysis  
    - Intelligent selector generation
    - Content validation and synthesis
    - Cost-optimized AI usage
    - Comprehensive error handling
    """
    
    def __init__(self, ai_client: AIAsyncClient, config: Optional[Dict[str, Any]] = None):
        # Configure agent with extraction-specific settings
        agent_config = AgentConfig(
            name="IntelligentContentExtractor",
            max_retries=3,
            rate_limit=8,  # Conservative for web scraping
            preferred_ai_provider="ollama",  # Use free provider for cost optimization
            task_complexity=TaskComplexity.COMPLEX,
            cache_ttl=3600,  # Cache results for 1 hour
            debug=config.get('debug', False) if config else False,
            timeout=45.0,  # Longer timeout for complex web operations
            min_confidence_score=0.6
        )
        
        super().__init__(agent_config, ai_client)
        
        # Initialize specialized components
        self.website_analyzer = WebsiteAnalyzer(ai_client, config)
        self.data_synthesizer = DataSynthesizer(ai_client, config)
        
        # Extraction configuration
        self.extraction_config = {
            'max_page_size': 15 * 1024 * 1024,  # 15MB limit
            'javascript_timeout': 30,
            'network_timeout': 30,
            'retry_failed_requests': True,
            'respect_robots_txt': True,
            'user_agent': 'AI-Ultimate-Service-Aggregator/1.0 (Business Research Bot)',
            'concurrent_extractions': 3
        }
        
        # Performance tracking
        self.extraction_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'average_extraction_time': 0.0,
            'total_ai_calls': 0,
            'total_cost': 0.0,
            'confidence_distribution': {},
            'error_types': {}
        }
        
        # Content type handlers
        self.content_handlers = {
            'company_info': self._extract_company_info,
            'services': self._extract_services_info,
            'pricing': self._extract_pricing_info,
            'contact': self._extract_contact_info,
            'about': self._extract_about_info
        }

    async def _setup_agent(self) -> None:
        """Initialize extraction agent and sub-components"""
        try:
            # Initialize sub-components
            await self.website_analyzer.initialize()
            await self.data_synthesizer.initialize()
            
            # Test AI connectivity with preferred free providers
            await self._test_ai_providers()
            
            self.logger.info("IntelligentContentExtractor initialized successfully")
            
        except Exception as e:
            raise AgentError(self.config.name, f"Failed to initialize content extractor: {e}")

    async def _cleanup_agent(self) -> None:
        """Cleanup extraction agent resources"""
        try:
            # Cleanup sub-components
            await self.website_analyzer.cleanup()
            await self.data_synthesizer.cleanup()
            
            # Log final statistics
            await self.log_operation(
                "extractor_shutdown",
                {
                    'total_extractions': self.extraction_stats['total_extractions'],
                    'success_rate': self._calculate_success_rate(),
                    'total_cost': self.extraction_stats['total_cost']
                }
            )
            
            self.logger.info("IntelligentContentExtractor cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup warning for content extractor: {e}")

    async def extract_provider_data(
        self, 
        target: ExtractionTarget
    ) -> ExtractionResult:
        """
        Main extraction method - orchestrates complete extraction pipeline
        
        Args:
            target: Extraction target specification
            
        Returns:
            Comprehensive extraction result
        """
        operation_id = self.generate_operation_id("content_extraction")
        start_time = time.time()
        
        self.logger.info(f"🔍 Starting content extraction for {target.website_url}")
        
        try:
            # Update statistics
            self.extraction_stats['total_extractions'] += 1
            
            # Stage 1: Website Analysis
            self.logger.debug("Stage 1: Analyzing website structure and characteristics")
            website_analysis = await self.website_analyzer.analyze_website(
                target.website_url, 
                target.service_category
            )
            
            # Stage 2: AI-Guided Content Extraction  
            self.logger.debug("Stage 2: Extracting content with AI guidance")
            extracted_content = await self._extract_content_with_ai_guidance(
                target, website_analysis
            )
            
            # Stage 3: Data Synthesis and Validation
            self.logger.debug("Stage 3: Synthesizing and validating extracted data")
            synthesized_data = await self.data_synthesizer.synthesize_provider_data(
                extracted_content, target, website_analysis
            )
            
            # Stage 4: Competitive Analysis
            self.logger.debug("Stage 4: Performing competitive analysis")
            competitive_analysis = await self._perform_competitive_analysis(
                synthesized_data, target
            )
            
            # Stage 5: Quality Assessment
            self.logger.debug("Stage 5: Assessing content quality and confidence")
            quality_assessment = await self._assess_extraction_quality(
                synthesized_data, website_analysis, extracted_content
            )
            
            # Create final result
            extraction_time = time.time() - start_time
            result = ExtractionResult(
                provider_name=synthesized_data.get('provider_name', target.provider_name or 'Unknown'),
                website_url=target.website_url,
                extraction_target=target,
                business_info=synthesized_data.get('business_info', {}),
                services=synthesized_data.get('services', []),
                pricing_info=synthesized_data.get('pricing', {}),
                contact_details=synthesized_data.get('contact', {}),
                competitive_analysis=competitive_analysis,
                technical_details=website_analysis.get('technical_details', {}),
                website_quality_score=quality_assessment.get('website_quality_score', 0.5),
                extraction_confidence=quality_assessment.get('overall_confidence', 0.5),
                extraction_timestamp=time.time(),
                extraction_duration=extraction_time,
                ai_providers_used=self._get_used_ai_providers(),
                raw_data={
                    'website_analysis': website_analysis,
                    'extracted_content': extracted_content,
                    'quality_assessment': quality_assessment
                }
            )
            
            # Update success statistics
            self.extraction_stats['successful_extractions'] += 1
            self._update_confidence_distribution(result.extraction_confidence)
            
            # Cache successful result
            cache_key = f"extraction_{target.website_url}_{target.service_category}"
            await self.cache_operation_result(cache_key, result)
            
            self.logger.info(
                f"✅ Content extraction completed for {target.website_url} "
                f"(confidence: {result.extraction_confidence:.2f}, duration: {extraction_time:.1f}s)"
            )
            
            return result
            
        except Exception as e:
            # Update failure statistics
            self.extraction_stats['failed_extractions'] += 1
            self._update_error_statistics(str(e))
            
            self.logger.error(f"❌ Content extraction failed for {target.website_url}: {e}")
            raise AgentError(self.config.name, f"Content extraction failed: {e}")

    async def _extract_content_with_ai_guidance(
        self, 
        target: ExtractionTarget, 
        website_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract content using AI-guided strategies based on website analysis
        
        Args:
            target: Extraction target
            website_analysis: Website analysis results
            
        Returns:
            Extracted content organized by type
        """
        extracted_content = {}
        
        # Get AI-recommended extraction strategy
        extraction_strategy = await self._generate_extraction_strategy(target, website_analysis)
        
        # Execute extraction for each content type
        for content_type in target.expected_content_types:
            if content_type in self.content_handlers:
                try:
                    content_data = await self.content_handlers[content_type](
                        target, website_analysis, extraction_strategy
                    )
                    extracted_content[content_type] = content_data
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract {content_type}: {e}")
                    extracted_content[content_type] = {}
        
        return extracted_content

    async def _generate_extraction_strategy(
        self, 
        target: ExtractionTarget, 
        website_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate AI-guided extraction strategy based on website characteristics
        
        Args:
            target: Extraction target
            website_analysis: Website analysis results
            
        Returns:
            Extraction strategy with selectors and methods
        """
        strategy_prompt = f"""
        Generate an intelligent extraction strategy for this website:
        
        Website: {target.website_url}
        Service Category: {target.service_category}
        Website Analysis: {json.dumps(website_analysis.get('structure_analysis', {}), indent=2)[:2000]}
        
        Based on the website structure and content patterns, recommend:
        
        1. **Extraction Methods**: Which methods to use for different content types
        2. **CSS Selectors**: Specific selectors for key content areas
        3. **Extraction Order**: Optimal order for content extraction
        4. **Fallback Strategies**: Alternative approaches if primary methods fail
        5. **Content Validation**: How to validate extracted content quality
        
        Return JSON with extraction strategy:
        {{
          "extraction_methods": {{
            "company_info": {{
              "primary_method": "playwright/requests/api",
              "selectors": ["selector1", "selector2"],
              "fallback_selectors": ["fallback1", "fallback2"],
              "validation_rules": ["rule1", "rule2"]
            }},
            "services": {{
              "primary_method": "playwright/requests/api",
              "selectors": ["selector1", "selector2"],
              "content_patterns": ["pattern1", "pattern2"],
              "extraction_hints": ["hint1", "hint2"]
            }},
            "pricing": {{
              "primary_method": "playwright/requests/api", 
              "selectors": ["selector1", "selector2"],
              "price_patterns": ["$[0-9]+", "€[0-9]+"],
              "context_clues": ["monthly", "annual", "one-time"]
            }},
            "contact": {{
              "primary_method": "playwright/requests/api",
              "selectors": ["selector1", "selector2"],
              "email_patterns": ["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,}}"],
              "phone_patterns": ["\\+?[1-9]?[0-9]{{7,14}}"]
            }}
          }},
          "extraction_sequence": ["content_type1", "content_type2"],
          "optimization_hints": {{
            "use_javascript": true/false,
            "wait_for_dynamic_content": true/false,
            "follow_pagination": true/false,
            "respect_rate_limits": true/false
          }},
          "confidence_indicators": {{
            "high_confidence": ["indicator1", "indicator2"],
            "medium_confidence": ["indicator3", "indicator4"],
            "low_confidence": ["indicator5", "indicator6"]
          }}
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=strategy_prompt,
                provider="ollama",  # Use free provider for strategy generation
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2000
            )
            
            strategy = safe_json_parse(response, default={})
            
            if not strategy:
                # Provide fallback strategy
                strategy = self._get_fallback_extraction_strategy(target, website_analysis)
            
            return strategy
            
        except Exception as e:
            self.logger.warning(f"Strategy generation failed, using fallback: {e}")
            return self._get_fallback_extraction_strategy(target, website_analysis)

    async def _extract_company_info(
        self, 
        target: ExtractionTarget, 
        website_analysis: Dict[str, Any], 
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract company information using AI-guided approach"""
        
        company_selectors = strategy.get('extraction_methods', {}).get('company_info', {}).get('selectors', [
            'h1', '.company-name', '.brand-name', '.logo-text', 'title',
            '[class*="company"]', '[class*="brand"]', '.about-company'
        ])
        
        try:
            # Use website analyzer to extract company content
            company_content = await self.website_analyzer.extract_content_by_selectors(
                target.website_url, company_selectors
            )
            
            # AI analysis of company information
            company_analysis_prompt = f"""
            Extract and structure company information from this content:
            
            Website: {target.website_url}
            Content: {json.dumps(company_content, indent=2)[:3000]}
            Service Category: {target.service_category}
            
            Extract structured company information:
            {{
              "company_name": "official_company_name",
              "business_description": "comprehensive_description",
              "founding_year": "year_or_null",
              "company_size": "estimated_employee_count_range",
              "headquarters_location": "city_country",
              "certifications": ["cert1", "cert2"],
              "awards": ["award1", "award2"],
              "key_personnel": [{{
                "name": "person_name",
                "role": "job_title"
              }}],
              "business_model": "b2b/b2c/b2b2c/marketplace",
              "target_market": "market_description",
              "unique_selling_proposition": "key_differentiators"
            }}
            """
            
            response = await self.ask_ai(
                prompt=company_analysis_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1500
            )
            
            return safe_json_parse(response, default={})
            
        except Exception as e:
            self.logger.error(f"Company info extraction failed: {e}")
            return {}

    async def _extract_services_info(
        self, 
        target: ExtractionTarget, 
        website_analysis: Dict[str, Any], 
        strategy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract services information using AI-guided approach"""
        
        services_selectors = strategy.get('extraction_methods', {}).get('services', {}).get('selectors', [
            '.services', '.products', '.offerings', '.solutions',
            '[class*="service"]', '[class*="product"]', '.service-list'
        ])
        
        try:
            # Extract services content
            services_content = await self.website_analyzer.extract_content_by_selectors(
                target.website_url, services_selectors
            )
            
            # AI analysis of services
            services_analysis_prompt = f"""
            Extract and structure services information from this content:
            
            Website: {target.website_url}
            Content: {json.dumps(services_content, indent=2)[:4000]}
            Service Category: {target.service_category}
            
            Extract list of services offered:
            [{{
              "service_name": "specific_service_name",
              "service_description": "detailed_description",
              "service_category": "category_or_type",
              "delivery_method": "onsite/remote/hybrid/online",
              "target_audience": "who_its_for",
              "key_features": ["feature1", "feature2"],
              "pricing_mentioned": "price_info_if_mentioned",
              "service_level": "basic/standard/premium/enterprise"
            }}]
            
            Focus on extracting distinct services, not generic marketing content.
            """
            
            response = await self.ask_ai(
                prompt=services_analysis_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2000
            )
            
            return safe_json_parse(response, default=[])
            
        except Exception as e:
            self.logger.error(f"Services info extraction failed: {e}")
            return []

    async def _extract_pricing_info(
        self, 
        target: ExtractionTarget, 
        website_analysis: Dict[str, Any], 
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract pricing information using AI-guided approach"""
        
        pricing_selectors = strategy.get('extraction_methods', {}).get('pricing', {}).get('selectors', [
            '.pricing', '.prices', '.plans', '.packages',
            '[class*="price"]', '[class*="plan"]', '.pricing-table'
        ])
        
        try:
            # Extract pricing content
            pricing_content = await self.website_analyzer.extract_content_by_selectors(
                target.website_url, pricing_selectors
            )
            
            # AI analysis of pricing
            pricing_analysis_prompt = f"""
            Extract and structure pricing information from this content:
            
            Website: {target.website_url}
            Content: {json.dumps(pricing_content, indent=2)[:3000]}
            Service Category: {target.service_category}
            
            Extract pricing structure:
            {{
              "pricing_model": "fixed/hourly/subscription/tiered/custom/quote_based",
              "pricing_transparency": "transparent/partially_transparent/opaque",
              "currency": "USD/EUR/GBP/etc",
              "pricing_plans": [{{
                "plan_name": "plan_name",
                "price": "price_amount",
                "billing_cycle": "monthly/annual/one_time",
                "features_included": ["feature1", "feature2"],
                "target_segment": "individual/small_business/enterprise"
              }}],
              "additional_fees": [{{
                "fee_type": "setup/maintenance/support",
                "amount": "fee_amount",
                "description": "fee_description"
              }}],
              "pricing_notes": "additional_pricing_information",
              "free_trial": {{
                "available": true/false,
                "duration": "trial_duration",
                "limitations": "trial_limitations"
              }}
            }}
            """
            
            response = await self.ask_ai(
                prompt=pricing_analysis_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1500
            )
            
            return safe_json_parse(response, default={})
            
        except Exception as e:
            self.logger.error(f"Pricing info extraction failed: {e}")
            return {}

    async def _extract_contact_info(
        self, 
        target: ExtractionTarget, 
        website_analysis: Dict[str, Any], 
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract contact information using AI-guided approach"""
        
        contact_selectors = strategy.get('extraction_methods', {}).get('contact', {}).get('selectors', [
            '.contact', '.contact-info', '.footer', '.header',
            '[class*="contact"]', '[id*="contact"]', '.contact-details'
        ])
        
        try:
            # Extract contact content
            contact_content = await self.website_analyzer.extract_content_by_selectors(
                target.website_url, contact_selectors
            )
            
            # AI analysis of contact information
            contact_analysis_prompt = f"""
            Extract and structure contact information from this content:
            
            Website: {target.website_url}
            Content: {json.dumps(contact_content, indent=2)[:3000]}
            
            Extract contact details:
            {{
              "primary_email": "main_business_email",
              "secondary_emails": ["email1", "email2"],
              "primary_phone": "main_phone_number",
              "secondary_phones": ["phone1", "phone2"],
              "address": {{
                "street": "street_address",
                "city": "city_name",
                "state_province": "state_or_province",
                "postal_code": "postal_code",
                "country": "country_name"
              }},
              "business_hours": {{
                "monday_friday": "business_hours",
                "weekend": "weekend_hours",
                "timezone": "timezone_info"
              }},
              "social_media": [{{
                "platform": "facebook/twitter/linkedin",
                "url": "social_media_url"
              }}],
              "contact_methods": ["phone", "email", "contact_form", "live_chat"],
              "support_availability": "24_7/business_hours/limited",
              "response_time": "response_time_commitment"
            }}
            """
            
            response = await self.ask_ai(
                prompt=contact_analysis_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1200
            )
            
            return safe_json_parse(response, default={})
            
        except Exception as e:
            self.logger.error(f"Contact info extraction failed: {e}")
            return {}

    async def _extract_about_info(
        self, 
        target: ExtractionTarget, 
        website_analysis: Dict[str, Any], 
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract about/company story information"""
        
        about_selectors = [
            '.about', '.about-us', '.company-story', '.our-story',
            '[href*="about"]', '[class*="about"]', '.company-overview'
        ]
        
        try:
            # Extract about content
            about_content = await self.website_analyzer.extract_content_by_selectors(
                target.website_url, about_selectors
            )
            
            # AI analysis of about information
            about_analysis_prompt = f"""
            Extract company story and background from this content:
            
            Content: {json.dumps(about_content, indent=2)[:2000]}
            
            Extract about information:
            {{
              "company_story": "founding_story_and_mission",
              "mission_statement": "company_mission",
              "vision_statement": "company_vision", 
              "core_values": ["value1", "value2"],
              "company_culture": "culture_description",
              "milestones": [{{
                "year": "milestone_year",
                "achievement": "what_was_achieved"
              }}],
              "team_size": "estimated_team_size",
              "locations": ["location1", "location2"],
              "industry_experience": "years_of_experience"
            }}
            """
            
            response = await self.ask_ai(
                prompt=about_analysis_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1000
            )
            
            return safe_json_parse(response, default={})
            
        except Exception as e:
            self.logger.error(f"About info extraction failed: {e}")
            return {}

    async def _perform_competitive_analysis(
        self, 
        synthesized_data: Dict[str, Any], 
        target: ExtractionTarget
    ) -> Dict[str, Any]:
        """Perform competitive analysis based on extracted data"""
        
        competitive_prompt = f"""
        Analyze the competitive positioning of this provider:
        
        Provider: {synthesized_data.get('provider_name', 'Unknown')}
        Service Category: {target.service_category}
        Services: {json.dumps(synthesized_data.get('services', []), indent=2)[:2000]}
        Pricing: {json.dumps(synthesized_data.get('pricing', {}), indent=2)[:1000]}
        
        Provide competitive analysis:
        {{
          "market_positioning": "premium/mid_market/budget/niche",
          "competitive_advantages": ["advantage1", "advantage2"],
          "potential_weaknesses": ["weakness1", "weakness2"],
          "target_customer_segment": "segment_description",
          "differentiation_factors": ["factor1", "factor2"],
          "pricing_strategy": "competitive/premium/value/penetration",
          "market_opportunity": "market_opportunity_assessment",
          "competitive_threats": ["threat1", "threat2"]
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=competitive_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1200
            )
            
            return safe_json_parse(response, default={})
            
        except Exception as e:
            self.logger.error(f"Competitive analysis failed: {e}")
            return {}

    async def _assess_extraction_quality(
        self, 
        synthesized_data: Dict[str, Any], 
        website_analysis: Dict[str, Any], 
        extracted_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess the quality and confidence of the extraction"""
        
        # Calculate various quality metrics
        data_completeness = self._calculate_data_completeness(synthesized_data)
        content_consistency = self._calculate_content_consistency(extracted_content)
        website_quality = website_analysis.get('quality_score', 0.5)
        
        # AI-based quality assessment
        quality_prompt = f"""
        Assess the quality and reliability of this extracted data:
        
        Data Completeness: {data_completeness:.2f}
        Content Consistency: {content_consistency:.2f}
        Website Quality: {website_quality:.2f}
        
        Extracted Data Summary:
        - Provider Name: {synthesized_data.get('provider_name', 'Unknown')}
        - Services Count: {len(synthesized_data.get('services', []))}
        - Has Pricing: {bool(synthesized_data.get('pricing', {}))}
        - Has Contact: {bool(synthesized_data.get('contact', {}))}
        
        Assess extraction quality:
        {{
          "overall_confidence": 0.0-1.0,
          "data_reliability": "high/medium/low",
          "completeness_score": 0.0-1.0,
          "accuracy_indicators": ["indicator1", "indicator2"],
          "quality_issues": ["issue1", "issue2"],
          "improvement_suggestions": ["suggestion1", "suggestion2"],
          "website_quality_score": 0.0-1.0
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=quality_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.SIMPLE,
                max_tokens=800
            )
            
            quality_assessment = safe_json_parse(response, default={})
            
            # Combine AI assessment with calculated metrics
            quality_assessment.update({
                'calculated_completeness': data_completeness,
                'calculated_consistency': content_consistency,
                'website_quality_score': website_quality
            })
            
            return quality_assessment
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return {
                'overall_confidence': 0.3,
                'data_reliability': 'low',
                'completeness_score': data_completeness,
                'website_quality_score': website_quality
            }

    def _calculate_data_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        required_fields = ['provider_name', 'business_info', 'services', 'contact']
        present_fields = sum(1 for field in required_fields if data.get(field))
        return present_fields / len(required_fields)

    def _calculate_content_consistency(self, content: Dict[str, Any]) -> float:
        """Calculate content consistency score"""
        total_extractions = len(content)
        successful_extractions = sum(1 for v in content.values() if v)
        return successful_extractions / max(1, total_extractions)

    def _get_fallback_extraction_strategy(
        self, 
        target: ExtractionTarget, 
        website_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide fallback extraction strategy when AI generation fails"""
        return {
            "extraction_methods": {
                "company_info": {
                    "primary_method": "playwright",
                    "selectors": ["h1", ".company-name", "title"],
                    "fallback_selectors": [".brand-name", ".logo-text"]
                },
                "services": {
                    "primary_method": "playwright",
                    "selectors": [".services", ".products", ".offerings"],
                    "content_patterns": ["service", "product", "solution"]
                },
                "pricing": {
                    "primary_method": "playwright",
                    "selectors": [".pricing", ".prices", ".plans"],
                    "price_patterns": ["$", "€", "£", "price", "cost"]
                },
                "contact": {
                    "primary_method": "playwright",
                    "selectors": [".contact", ".footer", ".header"],
                    "email_patterns": ["@"],
                    "phone_patterns": ["phone", "tel", "+"]
                }
            },
            "extraction_sequence": ["company_info", "services", "contact", "pricing"],
            "optimization_hints": {
                "use_javascript": True,
                "wait_for_dynamic_content": True,
                "respect_rate_limits": True
            }
        }

    async def _test_ai_providers(self):
        """Test connectivity and performance of AI providers"""
        test_prompt = "Test connectivity. Respond with: OK"
        
        for provider in ["ollama", "huggingface"]:
            try:
                response = await self.ask_ai(
                    prompt=test_prompt,
                    provider=provider,
                    task_complexity=TaskComplexity.SIMPLE
                )
                self.logger.debug(f"AI provider {provider} test: SUCCESS")
                
            except Exception as e:
                self.logger.warning(f"AI provider {provider} test failed: {e}")

    def _get_used_ai_providers(self) -> List[str]:
        """Get list of AI providers used in current extraction"""
        # This would track providers used during extraction
        # For now, return the preferred provider
        return [self.config.preferred_ai_provider]

    def _update_confidence_distribution(self, confidence: float):
        """Update confidence score distribution tracking"""
        bucket = int(confidence * 10) / 10  # Round to nearest 0.1
        self.extraction_stats['confidence_distribution'][bucket] = \
            self.extraction_stats['confidence_distribution'].get(bucket, 0) + 1

    def _update_error_statistics(self, error_message: str):
        """Update error type statistics"""
        error_type = error_message.split(':')[0] if ':' in error_message else 'Unknown'
        self.extraction_stats['error_types'][error_type] = \
            self.extraction_stats['error_types'].get(error_type, 0) + 1

    def _calculate_success_rate(self) -> float:
        """Calculate extraction success rate"""
        total = self.extraction_stats['total_extractions']
        if total == 0:
            return 0.0
        return self.extraction_stats['successful_extractions'] / total * 100

    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extraction statistics"""
        return {
            'performance': {
                'total_extractions': self.extraction_stats['total_extractions'],
                'success_rate': self._calculate_success_rate(),
                'average_extraction_time': self.extraction_stats['average_extraction_time'],
                'total_ai_calls': self.extraction_stats['total_ai_calls'],
                'total_cost': self.extraction_stats['total_cost']
            },
            'quality_metrics': {
                'confidence_distribution': self.extraction_stats['confidence_distribution'],
                'error_distribution': self.extraction_stats['error_types']
            },
            'agent_status': self.get_agent_status()
        }

    async def batch_extract_providers(
        self, 
        targets: List[ExtractionTarget],
        max_concurrent: Optional[int] = None
    ) -> List[ExtractionResult]:
        """
        Extract data from multiple providers concurrently
        
        Args:
            targets: List of extraction targets
            max_concurrent: Maximum concurrent extractions
            
        Returns:
            List of extraction results
        """
        actual_max_concurrent = max_concurrent or self.extraction_config['concurrent_extractions']
        
        self.logger.info(f"🔄 Starting batch extraction for {len(targets)} targets")
        
        # Create extraction operations
        operations = []
        for target in targets:
            operations.append({
                'type': 'function_call',
                'function': self.extract_provider_data,
                'args': [target],
                'kwargs': {}
            })
        
        # Execute batch operations
        results = await self.batch_operations(operations, max_concurrent=actual_max_concurrent)
        
        # Process results
        extraction_results = []
        for result in results:
            if result['status'] == 'success':
                extraction_results.append(result['result'])
            else:
                self.logger.error(f"Batch extraction failed: {result.get('error', 'Unknown error')}")
        
        self.logger.info(f"✅ Batch extraction completed: {len(extraction_results)}/{len(targets)} successful")
        
        return extraction_results