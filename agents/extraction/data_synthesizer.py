# agents/extraction/data_synthesizer.py
"""
DataSynthesizer - AI-powered data synthesis and validation

This component handles the synthesis of extracted website content into
structured, validated provider data with confidence scoring and quality assessment.
"""

import asyncio
import logging
import json
import time
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urlparse
import hashlib

from ..base import BaseAgent, AgentConfig, AgentError, safe_json_parse, retry_with_backoff
from services.ai_async_client import AIAsyncClient, TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Definition of a data validation rule"""
    field_name: str
    rule_type: str  # required, format, length, pattern, custom
    rule_value: Any
    error_message: str
    severity: str  # error, warning, info


@dataclass
class ValidationResult:
    """Result of data validation"""
    field_name: str
    is_valid: bool
    confidence_score: float
    validation_errors: List[str]
    validation_warnings: List[str]
    suggested_corrections: List[str]


@dataclass
class SynthesisMetrics:
    """Metrics for synthesis operation"""
    total_fields_processed: int
    successful_extractions: int
    validation_errors: int
    validation_warnings: int
    overall_confidence: float
    processing_time: float
    ai_calls_made: int


class DataSynthesizer:
    """
    Advanced AI-powered data synthesis and validation
    
    Features:
    - Multi-stage data synthesis pipeline
    - AI-powered data validation and correction
    - Confidence scoring algorithms
    - Quality assessment and improvement suggestions
    - Standardized output format generation
    - Cost-optimized AI usage
    """
    
    def __init__(self, ai_client: AIAsyncClient, config: Optional[Dict[str, Any]] = None):
        self.ai_client = ai_client
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Synthesis configuration
        self.synthesis_config = {
            'confidence_threshold': 0.6,
            'validation_strictness': 'moderate',  # strict, moderate, lenient
            'ai_correction_enabled': True,
            'batch_processing': True,
            'max_correction_attempts': 2,
            'quality_enhancement': True
        }
        
        # Validation rules for different data types
        self.validation_rules = self._initialize_validation_rules()
        
        # Data standardization patterns
        self.standardization_patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'phone': re.compile(r'^[\+]?[1-9]?[\-\s]?[\(]?[0-9]{1,4}[\)]?[\-\s]?[0-9]{1,4}[\-\s]?[0-9]{1,9}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            'price': re.compile(r'[\$€£¥₹]\s*[0-9]+(?:[.,][0-9]+)*'),
            'year': re.compile(r'^(19|20)\d{2}$')
        }
        
        # Performance tracking
        self.synthesis_stats = {
            'total_syntheses': 0,
            'successful_syntheses': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0,
            'validation_success_rate': 0.0,
            'ai_correction_success_rate': 0.0,
            'quality_improvements': 0
        }

    async def initialize(self):
        """Initialize data synthesizer components"""
        try:
            # Test AI connectivity for synthesis operations
            await self._test_synthesis_capabilities()
            
            # Load custom validation rules if configured
            await self._load_custom_validation_rules()
            
            self.logger.info("DataSynthesizer initialized successfully")
            
        except Exception as e:
            raise AgentError("DataSynthesizer", f"Failed to initialize: {e}")

    async def cleanup(self):
        """Cleanup synthesizer resources"""
        try:
            # Save synthesis statistics
            await self._save_synthesis_metrics()
            
            self.logger.info("DataSynthesizer cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup warning for DataSynthesizer: {e}")

    async def synthesize_provider_data(
        self, 
        extracted_content: Dict[str, Any],
        extraction_target: Any,  # ExtractionTarget from content_extractor
        website_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main synthesis method - converts raw extracted content into structured provider data
        
        Args:
            extracted_content: Raw content extracted from website
            extraction_target: Original extraction target specification
            website_analysis: Website analysis results
            
        Returns:
            Synthesized and validated provider data
        """
        synthesis_start = time.time()
        self.synthesis_stats['total_syntheses'] += 1
        
        self.logger.info(f"🔧 Starting data synthesis for {extraction_target.website_url}")
        
        try:
            # Stage 1: AI-powered content analysis and structuring
            self.logger.debug("Stage 1: AI content analysis and structuring")
            structured_data = await self._analyze_and_structure_content(
                extracted_content, extraction_target, website_analysis
            )
            
            # Stage 2: Data validation and correction
            self.logger.debug("Stage 2: Data validation and correction")
            validated_data = await self._validate_and_correct_data(
                structured_data, extraction_target
            )
            
            # Stage 3: Quality enhancement and enrichment
            self.logger.debug("Stage 3: Quality enhancement and enrichment")
            enhanced_data = await self._enhance_data_quality(
                validated_data, extraction_target, website_analysis
            )
            
            # Stage 4: Confidence scoring and assessment
            self.logger.debug("Stage 4: Confidence scoring and assessment")
            confidence_assessment = await self._assess_data_confidence(
                enhanced_data, extracted_content, website_analysis
            )
            
            # Stage 5: Standardization and formatting
            self.logger.debug("Stage 5: Standardization and formatting")
            standardized_data = await self._standardize_output_format(
                enhanced_data, confidence_assessment, extraction_target
            )
            
            # Calculate synthesis metrics
            synthesis_time = time.time() - synthesis_start
            metrics = self._calculate_synthesis_metrics(
                standardized_data, synthesis_time
            )
            
            # Update success statistics
            self.synthesis_stats['successful_syntheses'] += 1
            self._update_synthesis_statistics(metrics)
            
            # Add synthesis metadata
            standardized_data['synthesis_metadata'] = {
                'synthesized_at': time.time(),
                'synthesis_duration': synthesis_time,
                'metrics': asdict(metrics),
                'validation_summary': self._get_validation_summary(validated_data),
                'quality_indicators': self._get_quality_indicators(enhanced_data)
            }
            
            self.logger.info(
                f"✅ Data synthesis completed for {extraction_target.website_url} "
                f"(confidence: {confidence_assessment.get('overall_confidence', 0.0):.2f}, "
                f"duration: {synthesis_time:.1f}s)"
            )
            
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"❌ Data synthesis failed for {extraction_target.website_url}: {e}")
            raise AgentError("DataSynthesizer", f"Synthesis failed: {e}")

    async def _analyze_and_structure_content(
        self, 
        extracted_content: Dict[str, Any],
        extraction_target: Any,
        website_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use AI to analyze and structure the extracted content
        
        Args:
            extracted_content: Raw extracted content
            extraction_target: Extraction target specification
            website_analysis: Website analysis results
            
        Returns:
            Structured data organized by content type
        """
        
        # Prepare content summary for AI analysis
        content_summary = {}
        total_content_length = 0
        
        for content_type, content_data in extracted_content.items():
            if content_data:
                # Summarize content for AI processing
                if isinstance(content_data, dict):
                    text_content = str(content_data)
                elif isinstance(content_data, list):
                    text_content = ' '.join(str(item) for item in content_data)
                else:
                    text_content = str(content_data)
                
                content_summary[content_type] = {
                    'content_length': len(text_content),
                    'sample_content': text_content[:500] + "..." if len(text_content) > 500 else text_content,
                    'data_type': type(content_data).__name__
                }
                total_content_length += len(text_content)
        
        # AI-powered content structuring
        structuring_prompt = f"""
        Analyze and structure website content for a {extraction_target.service_category} provider:
        
        Website: {extraction_target.website_url}
        Provider Name: {extraction_target.provider_name or 'Unknown'}
        Service Category: {extraction_target.service_category}
        Website Quality: {website_analysis.get('quality_score', 0.5):.2f}
        
        Extracted Content Summary:
        {json.dumps(content_summary, indent=2)[:4000]}
        
        Structure this content into organized provider information:
        
        {{
          "provider_name": {{
            "extracted_value": "official_company_name",
            "confidence": 0.0-1.0,
            "sources": ["content_type1", "content_type2"],
            "alternatives": ["alternative_name1", "alternative_name2"]
          }},
          "business_info": {{
            "description": {{
              "value": "comprehensive_business_description",
              "confidence": 0.0-1.0,
              "source": "content_type"
            }},
            "founding_year": {{
              "value": "year_or_null",
              "confidence": 0.0-1.0,
              "source": "content_type"
            }},
            "company_size": {{
              "value": "size_category",
              "confidence": 0.0-1.0,
              "indicators": ["employee_count", "revenue_indicators"]
            }},
            "specializations": {{
              "value": ["specialization1", "specialization2"],
              "confidence": 0.0-1.0,
              "evidence": ["supporting_evidence"]
            }},
            "certifications": {{
              "value": ["cert1", "cert2"],
              "confidence": 0.0-1.0,
              "verification_needed": true/false
            }}
          }},
          "services": {{
            "main_services": [{{
              "name": "service_name",
              "description": "service_description",
              "category": "service_category",
              "confidence": 0.0-1.0,
              "evidence_strength": "strong/moderate/weak"
            }}],
            "service_delivery": {{
              "methods": ["onsite", "remote", "hybrid"],
              "coverage_area": "geographic_coverage",
              "confidence": 0.0-1.0
            }}
          }},
          "pricing": {{
            "structure": {{
              "type": "fixed/hourly/subscription/tiered/custom",
              "transparency": "transparent/partial/opaque",
              "confidence": 0.0-1.0
            }},
            "pricing_data": [{{
              "service": "service_name",
              "price_range": "price_information",
              "billing_cycle": "monthly/annual/one_time",
              "confidence": 0.0-1.0
            }}],
            "additional_costs": [{{
              "type": "setup/maintenance/support",
              "description": "cost_description",
              "confidence": 0.0-1.0
            }}]
          }},
          "contact": {{
            "primary_contact": {{
              "email": "primary_email",
              "phone": "primary_phone",
              "confidence": 0.0-1.0
            }},
            "address": {{
              "full_address": "complete_address",
              "city": "city",
              "country": "country",
              "confidence": 0.0-1.0
            }},
            "business_hours": {{
              "schedule": "operating_hours",
              "timezone": "timezone_info",
              "confidence": 0.0-1.0
            }}
          }},
          "data_quality_assessment": {{
            "completeness": 0.0-1.0,
            "consistency": 0.0-1.0,
            "accuracy_indicators": ["indicator1", "indicator2"],
            "missing_critical_info": ["missing_field1", "missing_field2"]
          }}
        }}
        
        Focus on accuracy, provide confidence scores, and identify missing information.
        """
        
        try:
            response = await self.ai_client.ask(
                prompt=structuring_prompt,
                provider="ollama",  # Use free provider for structuring
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=3000
            )
            
            structured_data = safe_json_parse(response, default={})
            
            if not structured_data:
                # Provide fallback structuring
                structured_data = self._get_fallback_structure(extracted_content, extraction_target)
            
            return structured_data
            
        except Exception as e:
            self.logger.error(f"Content structuring failed: {e}")
            return self._get_fallback_structure(extracted_content, extraction_target)

    async def _validate_and_correct_data(
        self, 
        structured_data: Dict[str, Any],
        extraction_target: Any
    ) -> Dict[str, Any]:
        """
        Validate structured data and apply AI-powered corrections
        
        Args:
            structured_data: Structured data from AI analysis
            extraction_target: Extraction target specification
            
        Returns:
            Validated and corrected data
        """
        
        validation_results = {}
        corrected_data = structured_data.copy()
        
        # Validate each major data section
        for section_name, section_data in structured_data.items():
            if section_name in ['provider_name', 'business_info', 'services', 'pricing', 'contact']:
                validation_result = await self._validate_data_section(
                    section_name, section_data, extraction_target
                )
                validation_results[section_name] = validation_result
                
                # Apply corrections if needed and enabled
                if (self.synthesis_config['ai_correction_enabled'] and 
                    not validation_result.is_valid and 
                    validation_result.suggested_corrections):
                    
                    corrected_section = await self._apply_ai_corrections(
                        section_name, section_data, validation_result, extraction_target
                    )
                    if corrected_section:
                        corrected_data[section_name] = corrected_section
        
        # Add validation metadata
        corrected_data['validation_results'] = validation_results
        corrected_data['validation_summary'] = self._summarize_validation_results(validation_results)
        
        return corrected_data

    async def _validate_data_section(
        self, 
        section_name: str, 
        section_data: Dict[str, Any],
        extraction_target: Any
    ) -> ValidationResult:
        """
        Validate a specific data section
        
        Args:
            section_name: Name of the data section
            section_data: Data to validate
            extraction_target: Extraction context
            
        Returns:
            Validation result with errors and suggestions
        """
        
        errors = []
        warnings = []
        suggestions = []
        field_confidence = []
        
        # Apply validation rules based on section type
        if section_name == 'provider_name':
            if not section_data.get('extracted_value'):
                errors.append("Provider name is missing")
            elif len(section_data.get('extracted_value', '')) < 2:
                errors.append("Provider name is too short")
            else:
                field_confidence.append(section_data.get('confidence', 0.5))
        
        elif section_name == 'contact':
            # Validate email format
            email = section_data.get('primary_contact', {}).get('email', '')
            if email and not self.standardization_patterns['email'].match(email):
                errors.append(f"Invalid email format: {email}")
                suggestions.append("Check email format and correct if needed")
            
            # Validate phone format
            phone = section_data.get('primary_contact', {}).get('phone', '')
            if phone and not self.standardization_patterns['phone'].match(phone):
                warnings.append(f"Phone format may be invalid: {phone}")
                suggestions.append("Standardize phone number format")
            
            # Check for missing critical contact info
            if not email and not phone:
                errors.append("No primary contact method found")
                suggestions.append("Search for contact information in footer or contact pages")
        
        elif section_name == 'services':
            services = section_data.get('main_services', [])
            if not services:
                errors.append("No services identified")
                suggestions.append("Look for service descriptions in main content areas")
            else:
                for service in services:
                    if service.get('confidence', 0) < 0.3:
                        warnings.append(f"Low confidence service: {service.get('name', 'Unknown')}")
                    field_confidence.append(service.get('confidence', 0.5))
        
        elif section_name == 'pricing':
            pricing_structure = section_data.get('structure', {})
            if not pricing_structure.get('type'):
                warnings.append("Pricing structure not clearly identified")
                suggestions.append("Check for pricing pages or service cost information")
        
        # Calculate overall validation confidence
        overall_confidence = sum(field_confidence) / len(field_confidence) if field_confidence else 0.3
        is_valid = len(errors) == 0 and overall_confidence >= self.synthesis_config['confidence_threshold']
        
        return ValidationResult(
            field_name=section_name,
            is_valid=is_valid,
            confidence_score=overall_confidence,
            validation_errors=errors,
            validation_warnings=warnings,
            suggested_corrections=suggestions
        )

    async def _apply_ai_corrections(
        self, 
        section_name: str, 
        section_data: Dict[str, Any],
        validation_result: ValidationResult,
        extraction_target: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Apply AI-powered corrections to validation errors
        
        Args:
            section_name: Name of data section
            section_data: Original section data
            validation_result: Validation results with errors
            extraction_target: Extraction context
            
        Returns:
            Corrected section data or None if correction failed
        """
        
        correction_prompt = f"""
        Correct validation errors in {section_name} data for {extraction_target.service_category} provider:
        
        Original Data: {json.dumps(section_data, indent=2)[:1500]}
        
        Validation Errors: {validation_result.validation_errors}
        Validation Warnings: {validation_result.validation_warnings}
        Suggested Corrections: {validation_result.suggested_corrections}
        
        Apply corrections to fix the identified issues:
        
        1. **Address Validation Errors**: Fix critical data issues
        2. **Improve Data Quality**: Enhance accuracy and completeness
        3. **Standardize Formats**: Apply proper formatting standards
        4. **Increase Confidence**: Provide more reliable data extraction
        
        Return corrected {section_name} data in the same format:
        {{
          // Corrected section data with same structure
          // Include confidence scores for corrected fields
          // Add correction_applied: true for modified fields
        }}
        
        Only apply corrections that improve data quality and accuracy.
        """
        
        try:
            response = await self.ai_client.ask(
                prompt=correction_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1500
            )
            
            corrected_data = safe_json_parse(response, default=None)
            
            if corrected_data:
                # Validate that corrections actually improve the data
                if await self._validate_corrections(section_data, corrected_data, validation_result):
                    self.synthesis_stats['ai_correction_success_rate'] += 1
                    return corrected_data
            
            return None
            
        except Exception as e:
            self.logger.warning(f"AI correction failed for {section_name}: {e}")
            return None

    async def _enhance_data_quality(
        self, 
        validated_data: Dict[str, Any],
        extraction_target: Any,
        website_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance data quality through AI analysis and enrichment
        
        Args:
            validated_data: Validated data
            extraction_target: Extraction target
            website_analysis: Website analysis results
            
        Returns:
            Enhanced data with quality improvements
        """
        
        if not self.synthesis_config['quality_enhancement']:
            return validated_data
        
        enhancement_prompt = f"""
        Enhance the quality and completeness of this provider data:
        
        Provider: {validated_data.get('provider_name', {}).get('extracted_value', 'Unknown')}
        Service Category: {extraction_target.service_category}
        Website Quality: {website_analysis.get('quality_score', 0.5):.2f}
        
        Current Data Quality:
        - Completeness: {validated_data.get('data_quality_assessment', {}).get('completeness', 0.5):.2f}
        - Consistency: {validated_data.get('data_quality_assessment', {}).get('consistency', 0.5):.2f}
        
        Provider Data: {json.dumps(validated_data, indent=2)[:3000]}
        
        Enhance data quality by:
        
        1. **Fill Information Gaps**: Infer missing information from available data
        2. **Improve Descriptions**: Make descriptions more comprehensive and professional
        3. **Standardize Terminology**: Use industry-standard terms and categories
        4. **Add Context**: Provide additional context and insights
        5. **Quality Scoring**: Improve confidence scores based on evidence strength
        
        Return enhanced data with quality improvements:
        {{
          // Same structure as input data
          // Enhanced descriptions and filled gaps
          // Improved confidence scores
          // Added "enhancement_applied" flag for modified fields
          // Include "quality_improvements" summary
        }}
        
        Focus on accuracy and maintain data integrity while enhancing quality.
        """
        
        try:
            response = await self.ai_client.ask(
                prompt=enhancement_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2500
            )
            
            enhanced_data = safe_json_parse(response, default=validated_data)
            
            # Track quality improvements
            if enhanced_data != validated_data:
                self.synthesis_stats['quality_improvements'] += 1
            
            return enhanced_data
            
        except Exception as e:
            self.logger.warning(f"Quality enhancement failed: {e}")
            return validated_data

    async def _assess_data_confidence(
        self, 
        enhanced_data: Dict[str, Any],
        extracted_content: Dict[str, Any],
        website_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess overall confidence in the synthesized data
        
        Args:
            enhanced_data: Enhanced provider data
            extracted_content: Original extracted content
            website_analysis: Website analysis results
            
        Returns:
            Comprehensive confidence assessment
        """
        
        # Calculate section-wise confidence scores
        section_confidences = {}
        
        for section_name in ['provider_name', 'business_info', 'services', 'pricing', 'contact']:
            if section_name in enhanced_data:
                section_confidence = self._calculate_section_confidence(
                    enhanced_data[section_name], section_name
                )
                section_confidences[section_name] = section_confidence
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            section_confidences, website_analysis, extracted_content
        )
        
        # Generate confidence factors
        confidence_factors = self._identify_confidence_factors(
            enhanced_data, website_analysis, extracted_content
        )
        
        return {
            'overall_confidence': overall_confidence,
            'section_confidences': section_confidences,
            'confidence_factors': confidence_factors,
            'reliability_assessment': self._assess_reliability(overall_confidence),
            'improvement_suggestions': self._generate_improvement_suggestions(
                section_confidences, confidence_factors
            )
        }

    async def _standardize_output_format(
        self, 
        enhanced_data: Dict[str, Any],
        confidence_assessment: Dict[str, Any],
        extraction_target: Any
    ) -> Dict[str, Any]:
        """
        Standardize data into the final output format
        
        Args:
            enhanced_data: Enhanced provider data
            confidence_assessment: Confidence assessment results
            extraction_target: Extraction target
            
        Returns:
            Standardized provider data ready for database storage
        """
        
        # Extract and clean core data
        provider_name = self._extract_clean_value(
            enhanced_data.get('provider_name', {}), 'extracted_value', 'Unknown'
        )
        
        # Business information
        business_info = self._standardize_business_info(
            enhanced_data.get('business_info', {})
        )
        
        # Services information
        services = self._standardize_services_info(
            enhanced_data.get('services', {})
        )
        
        # Pricing information
        pricing = self._standardize_pricing_info(
            enhanced_data.get('pricing', {})
        )
        
        # Contact information
        contact = self._standardize_contact_info(
            enhanced_data.get('contact', {})
        )
        
        # Create standardized output
        standardized_data = {
            'provider_name': provider_name,
            'business_info': business_info,
            'services': services,
            'pricing': pricing,
            'contact': contact,
            'confidence': confidence_assessment['overall_confidence'],
            'data_quality': {
                'completeness': self._calculate_completeness(enhanced_data),
                'accuracy': confidence_assessment['overall_confidence'],
                'consistency': self._calculate_consistency(enhanced_data),
                'reliability': confidence_assessment['reliability_assessment']
            },
            'extraction_metadata': {
                'target_url': extraction_target.website_url,
                'service_category': extraction_target.service_category,
                'extraction_depth': extraction_target.extraction_depth,
                'synthesis_version': '1.0.0'
            }
        }
        
        return standardized_data

    # Utility and helper methods
    
    def _initialize_validation_rules(self) -> Dict[str, List[ValidationRule]]:
        """Initialize validation rules for different data types"""
        return {
            'provider_name': [
                ValidationRule('extracted_value', 'required', True, 'Provider name is required', 'error'),
                ValidationRule('extracted_value', 'length', {'min': 2, 'max': 100}, 'Provider name length invalid', 'error'),
                ValidationRule('confidence', 'range', {'min': 0.3}, 'Low confidence in provider name', 'warning')
            ],
            'contact_email': [
                ValidationRule('email', 'format', 'email', 'Invalid email format', 'error'),
                ValidationRule('email', 'required', False, 'Email recommended for contact', 'warning')
            ],
            'contact_phone': [
                ValidationRule('phone', 'format', 'phone', 'Invalid phone format', 'warning'),
                ValidationRule('phone', 'required', False, 'Phone recommended for contact', 'warning')
            ],
            'services': [
                ValidationRule('main_services', 'required', True, 'Services list is required', 'error'),
                ValidationRule('main_services', 'length', {'min': 1}, 'At least one service required', 'error')
            ]
        }

    def _extract_clean_value(self, data_dict: Dict[str, Any], key: str, default: str) -> str:
        """Extract and clean a value from data dictionary"""
        if isinstance(data_dict, dict):
            value = data_dict.get(key, default)
        else:
            value = str(data_dict) if data_dict else default
        
        return str(value).strip() if value else default

    def _standardize_business_info(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize business information section"""
        return {
            'description': self._extract_clean_value(business_data.get('description', {}), 'value', ''),
            'founding_year': self._extract_clean_value(business_data.get('founding_year', {}), 'value', None),
            'company_size': self._extract_clean_value(business_data.get('company_size', {}), 'value', 'Unknown'),
            'specializations': business_data.get('specializations', {}).get('value', []),
            'certifications': business_data.get('certifications', {}).get('value', [])
        }

    def _standardize_services_info(self, services_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Standardize services information section"""
        main_services = services_data.get('main_services', [])
        
        standardized_services = []
        for service in main_services:
            if isinstance(service, dict):
                standardized_services.append({
                    'name': service.get('name', 'Unknown Service'),
                    'description': service.get('description', ''),
                    'category': service.get('category', 'General'),
                    'confidence': service.get('confidence', 0.5)
                })
        
        return standardized_services

    def _standardize_pricing_info(self, pricing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize pricing information section"""
        structure = pricing_data.get('structure', {})
        
        return {
            'structure': structure.get('type', 'unknown'),
            'transparency': structure.get('transparency', 'unknown'),
            'pricing_data': pricing_data.get('pricing_data', []),
            'additional_costs': pricing_data.get('additional_costs', [])
        }

    def _standardize_contact_info(self, contact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize contact information section"""
        primary_contact = contact_data.get('primary_contact', {})
        address = contact_data.get('address', {})
        
        return {
            'primary_email': primary_contact.get('email', ''),
            'primary_phone': primary_contact.get('phone', ''),
            'address': {
                'full_address': address.get('full_address', ''),
                'city': address.get('city', ''),
                'country': address.get('country', '')
            },
            'business_hours': contact_data.get('business_hours', {}).get('schedule', '')
        }

    def _calculate_section_confidence(self, section_data: Dict[str, Any], section_name: str) -> float:
        """Calculate confidence score for a data section"""
        if not section_data:
            return 0.0
        
        confidence_values = []
        
        # Extract confidence values from the section
        for key, value in section_data.items():
            if isinstance(value, dict) and 'confidence' in value:
                confidence_values.append(value['confidence'])
            elif key == 'confidence':
                confidence_values.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and 'confidence' in item:
                        confidence_values.append(item['confidence'])
        
        if confidence_values:
            return sum(confidence_values) / len(confidence_values)
        else:
            # Fallback confidence based on data completeness
            non_empty_fields = sum(1 for v in section_data.values() if v)
            total_fields = len(section_data)
            return non_empty_fields / max(1, total_fields) * 0.7  # Max 0.7 without explicit confidence

    def _calculate_overall_confidence(
        self, 
        section_confidences: Dict[str, float],
        website_analysis: Dict[str, Any],
        extracted_content: Dict[str, Any]
    ) -> float:
        """Calculate overall data confidence score"""
        
        # Weight different sections by importance
        section_weights = {
            'provider_name': 0.25,
            'contact': 0.25,
            'services': 0.20,
            'business_info': 0.15,
            'pricing': 0.15
        }
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for section, confidence in section_confidences.items():
            weight = section_weights.get(section, 0.1)
            weighted_confidence += confidence * weight
            total_weight += weight
        
        base_confidence = weighted_confidence / max(0.1, total_weight)
        
        # Adjust based on website quality
        website_quality = website_analysis.get('quality_score', 0.5)
        quality_adjustment = (website_quality - 0.5) * 0.2  # +/- 10% based on website quality
        
        # Adjust based on content richness
        content_richness = len([v for v in extracted_content.values() if v]) / max(1, len(extracted_content))
        richness_adjustment = (content_richness - 0.5) * 0.1  # +/- 5% based on content richness
        
        final_confidence = base_confidence + quality_adjustment + richness_adjustment
        
        return max(0.0, min(1.0, final_confidence))

    def _identify_confidence_factors(
        self, 
        enhanced_data: Dict[str, Any],
        website_analysis: Dict[str, Any],
        extracted_content: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Identify factors affecting confidence scores"""
        
        positive_factors = []
        negative_factors = []
        
        # Website quality factors
        website_quality = website_analysis.get('quality_score', 0.5)
        if website_quality > 0.7:
            positive_factors.append("High-quality professional website")
        elif website_quality < 0.3:
            negative_factors.append("Poor website quality affects data reliability")
        
        # Content completeness factors
        content_sections = len([v for v in extracted_content.values() if v])
        if content_sections >= 4:
            positive_factors.append("Comprehensive content coverage")
        elif content_sections <= 2:
            negative_factors.append("Limited content sections available")
        
        # Data validation factors
        validation_summary = enhanced_data.get('validation_summary', {})
        if validation_summary.get('error_count', 0) == 0:
            positive_factors.append("No validation errors found")
        else:
            negative_factors.append(f"{validation_summary.get('error_count', 0)} validation errors")
        
        # Contact information factors
        contact = enhanced_data.get('contact', {})
        if contact.get('primary_contact', {}).get('email') and contact.get('primary_contact', {}).get('phone'):
            positive_factors.append("Complete contact information available")
        elif not contact.get('primary_contact', {}).get('email'):
            negative_factors.append("Missing primary email contact")
        
        return {
            'positive_factors': positive_factors,
            'negative_factors': negative_factors
        }

    def _assess_reliability(self, overall_confidence: float) -> str:
        """Assess reliability level based on confidence score"""
        if overall_confidence >= 0.8:
            return "high"
        elif overall_confidence >= 0.6:
            return "medium"
        elif overall_confidence >= 0.4:
            return "low"
        else:
            return "very_low"

    def _generate_improvement_suggestions(
        self, 
        section_confidences: Dict[str, float],
        confidence_factors: Dict[str, List[str]]
    ) -> List[str]:
        """Generate suggestions for improving data quality"""
        suggestions = []
        
        # Low confidence sections
        for section, confidence in section_confidences.items():
            if confidence < 0.5:
                suggestions.append(f"Improve {section} data quality through additional sources")
        
        # Address negative factors
        for factor in confidence_factors.get('negative_factors', []):
            if "contact" in factor.lower():
                suggestions.append("Search for additional contact information sources")
            elif "validation" in factor.lower():
                suggestions.append("Review and correct validation errors")
            elif "content" in factor.lower():
                suggestions.append("Expand content extraction to additional page sections")
        
        return suggestions

    def _calculate_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        required_fields = [
            'provider_name', 'business_info.description', 'services', 'contact.primary_email'
        ]
        
        present_fields = 0
        for field_path in required_fields:
            if '.' in field_path:
                section, field = field_path.split('.', 1)
                if (section in data and 
                    isinstance(data[section], dict) and 
                    field in data[section] and 
                    data[section][field]):
                    present_fields += 1
            else:
                if field_path in data and data[field_path]:
                    present_fields += 1
        
        return present_fields / len(required_fields)

    def _calculate_consistency(self, data: Dict[str, Any]) -> float:
        """Calculate data consistency score"""
        # Simple consistency check - could be enhanced with more sophisticated logic
        inconsistencies = 0
        total_checks = 0
        
        # Check provider name consistency across sections
        provider_name = data.get('provider_name', '')
        if isinstance(provider_name, dict):
            provider_name = provider_name.get('extracted_value', '')
        
        # Additional consistency checks could be added here
        total_checks += 1
        
        return 1.0 - (inconsistencies / max(1, total_checks))

    def _calculate_synthesis_metrics(
        self, 
        standardized_data: Dict[str, Any], 
        processing_time: float
    ) -> SynthesisMetrics:
        """Calculate metrics for the synthesis operation"""
        
        # Count processed fields
        total_fields = 0
        successful_extractions = 0
        
        for section_name, section_data in standardized_data.items():
            if section_name in ['provider_name', 'business_info', 'services', 'pricing', 'contact']:
                if isinstance(section_data, dict):
                    for field_name, field_value in section_data.items():
                        total_fields += 1
                        if field_value:
                            successful_extractions += 1
                elif isinstance(section_data, list):
                    total_fields += len(section_data)
                    successful_extractions += len([item for item in section_data if item])
                else:
                    total_fields += 1
                    if section_data:
                        successful_extractions += 1
        
        # Get validation metrics from data
        validation_summary = standardized_data.get('validation_summary', {})
        
        return SynthesisMetrics(
            total_fields_processed=total_fields,
            successful_extractions=successful_extractions,
            validation_errors=validation_summary.get('error_count', 0),
            validation_warnings=validation_summary.get('warning_count', 0),
            overall_confidence=standardized_data.get('confidence', 0.0),
            processing_time=processing_time,
            ai_calls_made=4  # Approximate number of AI calls in synthesis pipeline
        )

    def _summarize_validation_results(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Summarize validation results across all sections"""
        total_errors = sum(len(result.validation_errors) for result in validation_results.values())
        total_warnings = sum(len(result.validation_warnings) for result in validation_results.values())
        valid_sections = sum(1 for result in validation_results.values() if result.is_valid)
        
        return {
            'total_sections': len(validation_results),
            'valid_sections': valid_sections,
            'error_count': total_errors,
            'warning_count': total_warnings,
            'overall_valid': total_errors == 0,
            'validation_score': valid_sections / max(1, len(validation_results))
        }

    def _get_validation_summary(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get validation summary from validated data"""
        return validated_data.get('validation_summary', {
            'total_sections': 0,
            'valid_sections': 0,
            'error_count': 0,
            'warning_count': 0
        })

    def _get_quality_indicators(self, enhanced_data: Dict[str, Any]) -> List[str]:
        """Get quality indicators from enhanced data"""
        indicators = []
        
        if enhanced_data.get('data_quality_assessment', {}).get('completeness', 0) > 0.8:
            indicators.append("high_completeness")
        
        if enhanced_data.get('data_quality_assessment', {}).get('consistency', 0) > 0.8:
            indicators.append("high_consistency")
        
        # Add more quality indicators as needed
        
        return indicators

    def _get_fallback_structure(
        self, 
        extracted_content: Dict[str, Any], 
        extraction_target: Any
    ) -> Dict[str, Any]:
        """Provide fallback structure when AI structuring fails"""
        return {
            'provider_name': {
                'extracted_value': extraction_target.provider_name or 'Unknown',
                'confidence': 0.3,
                'source': 'fallback'
            },
            'business_info': {
                'description': {'value': '', 'confidence': 0.1},
                'specializations': {'value': [extraction_target.service_category], 'confidence': 0.5}
            },
            'services': {
                'main_services': [{'name': 'General Services', 'confidence': 0.2}]
            },
            'pricing': {'structure': {'type': 'unknown', 'confidence': 0.1}},
            'contact': {'primary_contact': {'email': '', 'phone': '', 'confidence': 0.1}},
            'data_quality_assessment': {
                'completeness': 0.2,
                'consistency': 0.3,
                'accuracy_indicators': ['fallback_structure_used']
            }
        }

    async def _validate_corrections(
        self, 
        original_data: Dict[str, Any], 
        corrected_data: Dict[str, Any],
        validation_result: ValidationResult
    ) -> bool:
        """Validate that corrections actually improve the data"""
        # Simple validation - could be enhanced with more sophisticated logic
        
        # Check if the corrected data has fewer obvious issues
        original_issues = len(validation_result.validation_errors)
        
        # Basic checks on corrected data
        if not corrected_data:
            return False
        
        # Check if required fields are now present
        if validation_result.field_name == 'provider_name':
            corrected_name = corrected_data.get('extracted_value', '')
            if not corrected_name or len(corrected_name) < 2:
                return False
        
        return True  # Assume correction is valid if basic checks pass

    async def _test_synthesis_capabilities(self):
        """Test AI capabilities for synthesis operations"""
        test_prompt = "Test synthesis capability. Return: {'test': 'success'}"
        
        try:
            response = await self.ai_client.ask(
                prompt=test_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.SIMPLE
            )
            
            result = safe_json_parse(response, default={})
            if result.get('test') == 'success':
                self.logger.debug("Synthesis capability test: SUCCESS")
            else:
                self.logger.warning("Synthesis capability test: PARTIAL")
                
        except Exception as e:
            self.logger.warning(f"Synthesis capability test failed: {e}")

    async def _load_custom_validation_rules(self):
        """Load custom validation rules if configured"""
        # Placeholder for loading custom validation rules from configuration
        # This could be enhanced to load rules from external configuration files
        pass

    async def _save_synthesis_metrics(self):
        """Save synthesis performance metrics"""
        try:
            metrics_summary = {
                'total_syntheses': self.synthesis_stats['total_syntheses'],
                'success_rate': (
                    self.synthesis_stats['successful_syntheses'] / 
                    max(1, self.synthesis_stats['total_syntheses'])
                ) * 100,
                'average_confidence': self.synthesis_stats['average_confidence'],
                'quality_improvements': self.synthesis_stats['quality_improvements']
            }
            
            self.logger.info(f"Synthesis metrics: {metrics_summary}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save synthesis metrics: {e}")

    def _update_synthesis_statistics(self, metrics: SynthesisMetrics):
        """Update synthesis statistics with new metrics"""
        # Update average confidence
        current_count = self.synthesis_stats['successful_syntheses']
        if current_count == 1:
            self.synthesis_stats['average_confidence'] = metrics.overall_confidence
        else:
            current_avg = self.synthesis_stats['average_confidence']
            self.synthesis_stats['average_confidence'] = (
                (current_avg * (current_count - 1) + metrics.overall_confidence) / current_count
            )
        
        # Update average processing time
        if current_count == 1:
            self.synthesis_stats['average_processing_time'] = metrics.processing_time
        else:
            current_avg = self.synthesis_stats['average_processing_time']
            self.synthesis_stats['average_processing_time'] = (
                (current_avg * (current_count - 1) + metrics.processing_time) / current_count
            )

    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive synthesis statistics"""
        return {
            'performance_metrics': self.synthesis_stats,
            'configuration': self.synthesis_config,
            'validation_rules_count': sum(len(rules) for rules in self.validation_rules.values())
        }