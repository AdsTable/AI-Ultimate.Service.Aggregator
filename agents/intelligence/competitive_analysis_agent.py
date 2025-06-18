# agents/intelligence/competitive_analysis_agent.py
"""
CompetitiveAnalysisAgent - Advanced competitive intelligence and analysis

This agent specializes in comprehensive competitive analysis, including competitor
identification, positioning analysis, strategy assessment, and competitive threats monitoring.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re
import hashlib

from ..base import BaseAgent, AgentConfig, AgentError, safe_json_parse, retry_with_backoff
from services.ai_async_client import AIAsyncClient, TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class CompetitorProfile:
    """Comprehensive competitor profile data structure"""
    competitor_id: str
    company_name: str
    website_url: str
    business_model: str
    market_position: str
    
    # Core characteristics
    size_category: str  # startup, small, medium, large, enterprise
    geographic_coverage: List[str]
    service_offerings: List[Dict[str, Any]]
    pricing_strategy: Dict[str, Any]
    
    # Competitive positioning
    value_proposition: str
    competitive_advantages: List[str]
    weaknesses: List[str]
    target_customers: List[str]
    
    # Performance indicators
    market_share_estimate: float
    growth_trajectory: str
    financial_health: str
    innovation_capacity: str
    
    # Intelligence metadata
    last_analyzed: float
    analysis_confidence: float
    data_sources: List[str]


@dataclass
class CompetitiveAnalysisReport:
    """Comprehensive competitive analysis report"""
    analysis_timestamp: float
    target_market: str
    service_category: str
    analysis_scope: str
    
    # Competitor landscape
    identified_competitors: List[CompetitorProfile]
    competitive_landscape_summary: Dict[str, Any]
    
    # Strategic analysis
    competitive_positioning_map: Dict[str, Any]
    competitive_gaps: List[Dict[str, Any]]
    white_space_opportunities: List[Dict[str, Any]]
    
    # Threat assessment
    competitive_threats: List[Dict[str, Any]]
    emerging_competitors: List[Dict[str, Any]]
    disruption_risks: List[Dict[str, Any]]
    
    # Strategic recommendations
    positioning_recommendations: List[Dict[str, Any]]
    competitive_strategies: List[Dict[str, Any]]
    monitoring_priorities: List[str]
    
    # Quality metrics
    analysis_confidence: float
    competitors_analyzed: int
    data_completeness: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization"""
        return asdict(self)


class CompetitiveAnalysisAgent(BaseAgent):
    """
    Advanced competitive analysis agent for comprehensive competitive intelligence
    
    Features:
    - Automated competitor identification and profiling
    - Competitive positioning analysis and mapping
    - Strategic gap analysis and opportunity identification
    - Competitive threat assessment and monitoring
    - Real-time competitive intelligence updates
    - Strategic positioning recommendations
    """
    
    def __init__(self, ai_client: AIAsyncClient, config: Optional[Dict[str, Any]] = None):
        # Configure agent with competitive analysis specific settings
        agent_config = AgentConfig(
            name="CompetitiveAnalysisAgent",
            max_retries=3,
            rate_limit=12,  # Moderate rate for competitive analysis
            preferred_ai_provider="ollama",  # Cost optimization with free provider
            task_complexity=TaskComplexity.COMPLEX,
            cache_ttl=3600,  # 1 hour cache for competitive data
            debug=config.get('debug', False) if config else False,
            timeout=50.0,  # Extended timeout for complex analysis
            min_confidence_score=0.65
        )
        
        super().__init__(agent_config, ai_client)
        
        # Competitive analysis configuration
        self.analysis_config = {
            'competitor_identification_depth': 'comprehensive',
            'positioning_analysis_enabled': True,
            'threat_assessment_enabled': True,
            'gap_analysis_enabled': True,
            'strategic_recommendations_enabled': True,
            'competitive_monitoring_enabled': True,
            'confidence_threshold': 0.6
        }
        
        # Competitive analysis frameworks
        self.analysis_frameworks = {
            'positioning_dimensions': [
                'price_value', 'quality_service', 'innovation_technology',
                'market_reach', 'customer_focus', 'brand_strength'
            ],
            'competitive_factors': [
                'product_quality', 'pricing_strategy', 'market_presence',
                'customer_service', 'innovation_capability', 'financial_strength',
                'operational_efficiency', 'brand_recognition'
            ],
            'threat_categories': [
                'direct_competition', 'substitute_products', 'new_entrants',
                'pricing_pressure', 'market_disruption', 'technology_obsolescence'
            ]
        }
        
        # Competitor identification patterns
        self.competitor_signals = {
            'direct_competitors': [
                'same_service_category', 'overlapping_target_market',
                'similar_value_proposition', 'comparable_pricing'
            ],
            'indirect_competitors': [
                'alternative_solutions', 'substitute_services',
                'different_approach_same_need', 'adjacent_markets'
            ],
            'emerging_competitors': [
                'new_business_models', 'technology_disruption',
                'startup_innovation', 'market_expansion'
            ]
        }
        
        # Performance tracking
        self.competitive_stats = {
            'total_analyses': 0,
            'competitors_identified': 0,
            'threats_detected': 0,
            'opportunities_found': 0,
            'recommendations_generated': 0,
            'analysis_accuracy_feedback': [],
            'average_analysis_time': 0.0
        }

    async def _setup_agent(self) -> None:
        """Initialize competitive analysis agent"""
        try:
            # Test AI capabilities for competitive analysis
            await self._test_competitive_analysis_capabilities()
            
            # Initialize competitive intelligence databases
            await self._initialize_competitive_databases()
            
            # Setup competitive monitoring frameworks
            await self._setup_monitoring_frameworks()
            
            self.logger.info("CompetitiveAnalysisAgent initialized successfully")
            
        except Exception as e:
            raise AgentError(self.config.name, f"Failed to initialize competitive analysis agent: {e}")

    async def _cleanup_agent(self) -> None:
        """Cleanup competitive analysis resources"""
        try:
            # Save competitive intelligence cache
            await self._save_competitive_intelligence()
            
            # Export competitive analysis metrics
            await self._export_competitive_metrics()
            
            self.logger.info("CompetitiveAnalysisAgent cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup warning for competitive analysis agent: {e}")

    async def analyze_competitive_landscape(
        self, 
        target_market: str,
        service_category: str,
        analysis_scope: str = "comprehensive",
        known_competitors: Optional[List[str]] = None
    ) -> CompetitiveAnalysisReport:
        """
        Perform comprehensive competitive landscape analysis
        
        Args:
            target_market: Target market/country for analysis
            service_category: Service category to analyze
            analysis_scope: Scope of analysis (basic, standard, comprehensive)
            known_competitors: List of known competitors to include
            
        Returns:
            Comprehensive competitive analysis report
        """
        analysis_start = time.time()
        self.competitive_stats['total_analyses'] += 1
        
        self.logger.info(f"🎯 Starting competitive analysis for {service_category} in {target_market}")
        
        try:
            # Phase 1: Competitor Identification and Discovery
            self.logger.debug("Phase 1: Competitor identification and discovery")
            identified_competitors = await self._identify_competitors(
                target_market, service_category, known_competitors
            )
            
            # Phase 2: Competitor Profiling and Analysis
            self.logger.debug("Phase 2: Competitor profiling and analysis")
            competitor_profiles = await self._profile_competitors(
                identified_competitors, target_market, service_category
            )
            
            # Phase 3: Competitive Positioning Analysis
            self.logger.debug("Phase 3: Competitive positioning analysis")
            positioning_analysis = await self._analyze_competitive_positioning(
                competitor_profiles, service_category
            )
            
            # Phase 4: Gap Analysis and Opportunity Identification
            self.logger.debug("Phase 4: Gap analysis and opportunity identification")
            gap_analysis, opportunities = await self._analyze_competitive_gaps(
                competitor_profiles, positioning_analysis
            )
            
            # Phase 5: Threat Assessment and Risk Analysis
            self.logger.debug("Phase 5: Threat assessment and risk analysis")
            threat_assessment = await self._assess_competitive_threats(
                competitor_profiles, target_market, service_category
            )
            
            # Phase 6: Strategic Recommendations Generation
            self.logger.debug("Phase 6: Strategic recommendations generation")
            recommendations = await self._generate_strategic_recommendations(
                competitor_profiles, positioning_analysis, gap_analysis, threat_assessment
            )
            
            # Phase 7: Competitive Intelligence Quality Assessment
            self.logger.debug("Phase 7: Quality assessment")
            quality_assessment = await self._assess_analysis_quality(
                competitor_profiles, positioning_analysis
            )
            
            # Create comprehensive competitive analysis report
            analysis_duration = time.time() - analysis_start
            
            report = CompetitiveAnalysisReport(
                analysis_timestamp=time.time(),
                target_market=target_market,
                service_category=service_category,
                analysis_scope=analysis_scope,
                identified_competitors=competitor_profiles,
                competitive_landscape_summary=self._summarize_competitive_landscape(competitor_profiles),
                competitive_positioning_map=positioning_analysis,
                competitive_gaps=gap_analysis,
                white_space_opportunities=opportunities,
                competitive_threats=threat_assessment['threats'],
                emerging_competitors=threat_assessment['emerging_competitors'],
                disruption_risks=threat_assessment['disruption_risks'],
                positioning_recommendations=recommendations['positioning'],
                competitive_strategies=recommendations['strategies'],
                monitoring_priorities=recommendations['monitoring_priorities'],
                analysis_confidence=quality_assessment['overall_confidence'],
                competitors_analyzed=len(competitor_profiles),
                data_completeness=quality_assessment['data_completeness']
            )
            
            # Update statistics
            self.competitive_stats['competitors_identified'] += len(competitor_profiles)
            self.competitive_stats['threats_detected'] += len(threat_assessment['threats'])
            self.competitive_stats['opportunities_found'] += len(opportunities)
            self.competitive_stats['recommendations_generated'] += len(recommendations['strategies'])
            self._update_analysis_time(analysis_duration)
            
            # Cache analysis results
            cache_key = f"competitive_analysis_{target_market}_{service_category}_{analysis_scope}"
            await self.cache_operation_result(cache_key, report)
            
            self.logger.info(
                f"✅ Competitive analysis completed for {service_category} in {target_market} "
                f"({len(competitor_profiles)} competitors, confidence: {report.analysis_confidence:.2f}, "
                f"duration: {analysis_duration:.1f}s)"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Competitive analysis failed: {e}")
            raise AgentError(self.config.name, f"Competitive analysis failed: {e}")

    async def _identify_competitors(
        self, 
        target_market: str,
        service_category: str,
        known_competitors: Optional[List[str]] = None
    ) -> List[str]:
        """
        Identify competitors using AI-powered analysis
        
        Args:
            target_market: Target market for analysis
            service_category: Service category
            known_competitors: Known competitors to include
            
        Returns:
            List of identified competitor names/companies
        """
        
        identification_prompt = f"""
        Identify key competitors in the {service_category} market in {target_market}:
        
        Market Context:
        - Service Category: {service_category}
        - Target Market: {target_market}
        - Known Competitors: {known_competitors or 'None specified'}
        
        Identify competitors across different categories:
        
        1. **Direct Competitors**:
           - Companies offering identical or very similar services
           - Targeting the same customer segments
           - Operating in the same geographic market
           - Similar value propositions and business models
        
        2. **Indirect Competitors**:
           - Companies offering alternative solutions to the same customer need
           - Different approaches to solving similar problems
           - Substitute services or products
           - Adjacent market players expanding into this space
        
        3. **Emerging Competitors**:
           - New entrants with innovative approaches
           - Startups disrupting traditional models
           - Technology companies expanding into this market
           - International players entering the market
        
        4. **Strategic Competitors**:
           - Market leaders setting industry standards
           - Companies with significant market influence
           - Players with strong competitive advantages
           - Potential acquisition targets or threats
        
        Focus on companies that are:
        - Legitimate businesses (not individuals or informal services)
        - Currently active in the market
        - Have significant market presence or growth potential
        - Represent competitive threats or opportunities
        
        Return competitor identification results:
        {{
          "direct_competitors": [{{
            "company_name": "competitor_name",
            "website": "company_website_if_known",
            "market_position": "leader/challenger/follower/niche",
            "competitive_threat_level": "high/medium/low",
            "identification_confidence": 0.0-1.0,
            "reasoning": "why_this_is_a_direct_competitor"
          }}],
          "indirect_competitors": [{{
            "company_name": "competitor_name",
            "alternative_approach": "how_they_address_same_need",
            "market_overlap": "high/medium/low",
            "substitution_risk": "high/medium/low",
            "identification_confidence": 0.0-1.0
          }}],
          "emerging_competitors": [{{
            "company_name": "competitor_name",
            "innovation_factor": "what_makes_them_disruptive",
            "growth_stage": "startup/scale_up/expanding",
            "disruption_potential": "high/medium/low",
            "identification_confidence": 0.0-1.0
          }}],
          "market_leaders": [{{
            "company_name": "leader_name",
            "leadership_factors": ["factor1", "factor2"],
            "market_influence": "high/medium/low",
            "competitive_moat": "strong/moderate/weak"
          }}],
          "identification_summary": {{
            "total_competitors_identified": "number",
            "market_concentration": "fragmented/moderate/concentrated",
            "competitive_intensity": "intense/moderate/low",
            "identification_confidence": 0.0-1.0
          }}
        }}
        
        Focus on companies with significant market presence or strategic importance.
        """
        
        try:
            response = await self.ask_ai(
                prompt=identification_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2500
            )
            
            identification_result = safe_json_parse(response, default={})
            
            # Extract all identified competitors
            all_competitors = []
            
            for category in ['direct_competitors', 'indirect_competitors', 'emerging_competitors', 'market_leaders']:
                competitors = identification_result.get(category, [])
                for competitor in competitors:
                    competitor_name = competitor.get('company_name', '').strip()
                    if competitor_name and competitor_name not in all_competitors:
                        all_competitors.append(competitor_name)
            
            # Add known competitors if not already identified
            if known_competitors:
                for known in known_competitors:
                    if known.strip() and known.strip() not in all_competitors:
                        all_competitors.append(known.strip())
            
            # Store detailed identification results for later use
            self._competitor_identification_details = identification_result
            
            self.logger.info(f"🔍 Identified {len(all_competitors)} competitors for analysis")
            
            return all_competitors
            
        except Exception as e:
            self.logger.error(f"Competitor identification failed: {e}")
            # Fallback to known competitors if provided
            return known_competitors or []

    async def _profile_competitors(
        self, 
        competitor_names: List[str],
        target_market: str,
        service_category: str
    ) -> List[CompetitorProfile]:
        """
        Create detailed profiles for identified competitors
        
        Args:
            competitor_names: List of competitor names to profile
            target_market: Target market context
            service_category: Service category context
            
        Returns:
            List of detailed competitor profiles
        """
        
        competitor_profiles = []
        
        for competitor_name in competitor_names:
            try:
                # Create detailed competitor profile using AI analysis
                profile = await self._create_competitor_profile(
                    competitor_name, target_market, service_category
                )
                
                if profile and profile.analysis_confidence >= self.analysis_config['confidence_threshold']:
                    competitor_profiles.append(profile)
                
                # Rate limiting between competitor analyses
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.warning(f"Failed to profile competitor {competitor_name}: {e}")
                continue
        
        self.logger.info(f"📊 Successfully profiled {len(competitor_profiles)} competitors")
        
        return competitor_profiles

    async def _create_competitor_profile(
        self, 
        competitor_name: str,
        target_market: str,
        service_category: str
    ) -> Optional[CompetitorProfile]:
        """
        Create detailed profile for a single competitor
        
        Args:
            competitor_name: Name of competitor to profile
            target_market: Target market context
            service_category: Service category context
            
        Returns:
            Detailed competitor profile or None if analysis fails
        """
        
        profiling_prompt = f"""
        Create comprehensive competitor profile for {competitor_name} in {service_category} market in {target_market}:
        
        Competitor: {competitor_name}
        Market Context: {service_category} services in {target_market}
        
        Analyze and profile this competitor across key dimensions:
        
        1. **Company Overview**:
           - Business model and approach
           - Company size and scale indicators
           - Geographic coverage and presence
           - Years in operation and market experience
        
        2. **Service Offerings**:
           - Core services and products
           - Service quality and delivery methods
           - Pricing strategy and models
           - Unique value propositions
        
        3. **Market Position**:
           - Market position and reputation
           - Customer segments and target markets
           - Brand strength and recognition
           - Market share indicators
        
        4. **Competitive Strengths**:
           - Key competitive advantages
           - Areas of excellence and differentiation
           - Resources and capabilities
           - Strategic assets and moats
        
        5. **Competitive Weaknesses**:
           - Areas of vulnerability
           - Service or capability gaps
           - Customer complaints or issues
           - Strategic limitations
        
        6. **Strategic Intelligence**:
           - Growth trajectory and ambitions
           - Innovation capacity and R&D
           - Financial health indicators
           - Strategic partnerships and alliances
        
        Return detailed competitor profile:
        {{
          "competitor_overview": {{
            "company_name": "{competitor_name}",
            "estimated_website": "company_website_if_identifiable",
            "business_model": "business_model_description",
            "company_size": "startup/small/medium/large/enterprise",
            "geographic_coverage": ["region1", "region2"],
            "market_experience": "years_in_market_estimate",
            "founding_context": "company_background_if_known"
          }},
          "service_portfolio": {{
            "core_services": [{{
              "service_name": "service_description",
              "service_category": "service_type",
              "quality_level": "premium/standard/budget",
              "delivery_method": "onsite/remote/hybrid"
            }}],
            "pricing_strategy": {{
              "pricing_model": "fixed/hourly/subscription/value_based",
              "price_positioning": "premium/competitive/discount",
              "pricing_transparency": "transparent/partial/opaque"
            }},
            "value_proposition": "unique_value_and_positioning"
          }},
          "market_positioning": {{
            "market_position": "leader/challenger/follower/niche_player",
            "target_customers": ["customer_segment1", "customer_segment2"],
            "brand_strength": "strong/moderate/weak/unknown",
            "market_share_estimate": 0.0-1.0,
            "reputation_indicators": ["indicator1", "indicator2"]
          }},
          "competitive_strengths": {{
            "key_advantages": ["advantage1", "advantage2"],
            "excellence_areas": ["area1", "area2"],
            "strategic_assets": ["asset1", "asset2"],
            "competitive_moats": ["moat1", "moat2"]
          }},
          "competitive_weaknesses": {{
            "vulnerability_areas": ["weakness1", "weakness2"],
            "service_gaps": ["gap1", "gap2"],
            "customer_issues": ["issue1", "issue2"],
            "strategic_limitations": ["limitation1", "limitation2"]
          }},
          "strategic_intelligence": {{
            "growth_trajectory": "rapid/steady/slow/declining",
            "innovation_capacity": "high/medium/low/unknown",
            "financial_health": "strong/stable/concerning/unknown",
            "strategic_direction": "expansion/consolidation/pivot/maintenance",
            "partnership_strategy": "active/selective/minimal/unknown"
          }},
          "competitive_assessment": {{
            "overall_threat_level": "high/medium/low",
            "competitive_intensity": "intense/moderate/limited",
            "differentiation_strength": "strong/moderate/weak",
            "market_influence": "high/medium/low"
          }},
          "profile_confidence": 0.0-1.0,
          "data_sources": ["ai_analysis", "market_knowledge", "business_intelligence"],
          "analysis_limitations": ["limitation1", "limitation2"]
        }}
        
        Focus on providing actionable competitive intelligence while being realistic about confidence levels.
        """
        
        try:
            response = await self.ask_ai(
                prompt=profiling_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2000
            )
            
            profile_data = safe_json_parse(response, default={})
            
            if not profile_data:
                return None
            
            # Extract profile information
            overview = profile_data.get('competitor_overview', {})
            services = profile_data.get('service_portfolio', {})
            positioning = profile_data.get('market_positioning', {})
            strengths = profile_data.get('competitive_strengths', {})
            weaknesses = profile_data.get('competitive_weaknesses', {})
            intelligence = profile_data.get('strategic_intelligence', {})
            assessment = profile_data.get('competitive_assessment', {})
            
            # Create competitor profile object
            profile = CompetitorProfile(
                competitor_id=self._generate_competitor_id(competitor_name),
                company_name=competitor_name,
                website_url=overview.get('estimated_website', ''),
                business_model=overview.get('business_model', 'Unknown'),
                market_position=positioning.get('market_position', 'unknown'),
                size_category=overview.get('company_size', 'unknown'),
                geographic_coverage=overview.get('geographic_coverage', []),
                service_offerings=services.get('core_services', []),
                pricing_strategy=services.get('pricing_strategy', {}),
                value_proposition=services.get('value_proposition', ''),
                competitive_advantages=strengths.get('key_advantages', []),
                weaknesses=weaknesses.get('vulnerability_areas', []),
                target_customers=positioning.get('target_customers', []),
                market_share_estimate=positioning.get('market_share_estimate', 0.0),
                growth_trajectory=intelligence.get('growth_trajectory', 'unknown'),
                financial_health=intelligence.get('financial_health', 'unknown'),
                innovation_capacity=intelligence.get('innovation_capacity', 'unknown'),
                last_analyzed=time.time(),
                analysis_confidence=profile_data.get('profile_confidence', 0.5),
                data_sources=profile_data.get('data_sources', ['ai_analysis'])
            )
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Competitor profiling failed for {competitor_name}: {e}")
            return None

    async def _analyze_competitive_positioning(
        self, 
        competitor_profiles: List[CompetitorProfile],
        service_category: str
    ) -> Dict[str, Any]:
        """
        Analyze competitive positioning and create positioning map
        
        Args:
            competitor_profiles: List of competitor profiles
            service_category: Service category context
            
        Returns:
            Competitive positioning analysis
        """
        
        # Prepare competitor data for positioning analysis
        competitors_summary = []
        for profile in competitor_profiles:
            competitors_summary.append({
                'name': profile.company_name,
                'position': profile.market_position,
                'size': profile.size_category,
                'value_prop': profile.value_proposition,
                'pricing': profile.pricing_strategy.get('price_positioning', 'unknown'),
                'strengths': profile.competitive_advantages[:3],  # Top 3 strengths
                'target_customers': profile.target_customers
            })
        
        positioning_prompt = f"""
        Analyze competitive positioning for {service_category} market:
        
        Competitor Profiles Summary:
        {json.dumps(competitors_summary, indent=2)[:3000]}
        
        Analyze competitive positioning across key dimensions:
        
        1. **Positioning Dimensions Analysis**:
           - Price vs Value positioning
           - Service Quality vs Market Reach
           - Innovation vs Stability
           - Specialization vs Generalization
        
        2. **Strategic Groups Identification**:
           - Group competitors by similar strategies
           - Identify positioning clusters
           - Analyze strategic group boundaries
           - Assess mobility barriers between groups
        
        3. **Competitive Positioning Map**:
           - Map competitors on key dimensions
           - Identify positioning gaps and white spaces
           - Analyze competitive overlaps
           - Assess differentiation levels
        
        4. **Market Segmentation Analysis**:
           - Customer segment targeting patterns
           - Segment overlap and competition
           - Underserved or overserved segments
           - Niche opportunities
        
        Return competitive positioning analysis:
        {{
          "positioning_dimensions": {{
            "primary_dimensions": [{{
              "dimension_name": "price_value/quality_reach/innovation_stability",
              "axis_description": "dimension_explanation",
              "competitive_spread": "concentrated/dispersed/polarized",
              "differentiation_opportunity": "high/medium/low"
            }}],
            "positioning_insights": ["insight1", "insight2"]
          }},
          "strategic_groups": [{{
            "group_name": "group_description",
            "members": ["company1", "company2"],
            "strategy_characteristics": ["characteristic1", "characteristic2"],
            "group_performance": "strong/moderate/weak",
            "mobility_barriers": ["barrier1", "barrier2"]
          }}],
          "positioning_map": {{
            "competitor_positions": [{{
              "competitor": "company_name",
              "x_axis_value": "position_on_primary_dimension",
              "y_axis_value": "position_on_secondary_dimension",
              "market_size": "relative_market_presence",
              "positioning_strength": "strong/moderate/weak"
            }}],
            "white_spaces": [{{
              "space_description": "positioning_gap_description",
              "opportunity_size": "large/medium/small",
              "access_difficulty": "easy/moderate/difficult"
            }}]
          }},
          "segmentation_analysis": {{
            "customer_segments": [{{
              "segment_name": "segment_description",
              "competitors_targeting": ["company1", "company2"],
              "competitive_intensity": "high/medium/low",
              "segment_attractiveness": "high/medium/low",
              "serving_quality": "overserved/well_served/underserved"
            }}],
            "segment_opportunities": ["opportunity1", "opportunity2"]
          }},
          "differentiation_analysis": {{
            "highly_differentiated": ["company1", "company2"],
            "moderately_differentiated": ["company3", "company4"],
            "low_differentiation": ["company5", "company6"],
            "differentiation_factors": ["factor1", "factor2"],
            "commoditization_risk": "high/medium/low"
          }},
          "positioning_recommendations": {{
            "optimal_positioning": "recommended_positioning_strategy",
            "differentiation_opportunities": ["opportunity1", "opportunity2"],
            "positioning_risks": ["risk1", "risk2"],
            "strategic_moves": ["move1", "move2"]
          }},
          "analysis_confidence": 0.0-1.0
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=positioning_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2500
            )
            
            positioning_analysis = safe_json_parse(response, default={})
            
            if not positioning_analysis:
                positioning_analysis = self._get_fallback_positioning_analysis(competitor_profiles)
            
            # Add positioning metadata
            positioning_analysis['analysis_metadata'] = {
                'analyzed_at': time.time(),
                'competitors_analyzed': len(competitor_profiles),
                'analysis_method': 'ai_competitive_positioning'
            }
            
            return positioning_analysis
            
        except Exception as e:
            self.logger.error(f"Competitive positioning analysis failed: {e}")
            return self._get_fallback_positioning_analysis(competitor_profiles)

    async def _analyze_competitive_gaps(
        self, 
        competitor_profiles: List[CompetitorProfile],
        positioning_analysis: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Analyze competitive gaps and identify opportunities
        
        Args:
            competitor_profiles: List of competitor profiles
            positioning_analysis: Competitive positioning analysis
            
        Returns:
            Tuple of competitive gaps and white space opportunities
        """
        
        gaps_prompt = f"""
        Analyze competitive gaps and opportunities based on competitor analysis:
        
        Competitors Count: {len(competitor_profiles)}
        Positioning Insights: {positioning_analysis.get('positioning_insights', [])}
        White Spaces: {positioning_analysis.get('positioning_map', {}).get('white_spaces', [])}
        
        Competitor Weaknesses Summary:
        {json.dumps([{'name': p.company_name, 'weaknesses': p.weaknesses} for p in competitor_profiles], indent=2)[:2000]}
        
        Identify competitive gaps and opportunities:
        
        1. **Service and Capability Gaps**:
           - Unmet customer needs
           - Service quality inconsistencies
           - Technology or innovation gaps
           - Geographic coverage gaps
        
        2. **Market Positioning Gaps**:
           - Underserved customer segments
           - Price-value positioning gaps
           - Brand positioning weaknesses
           - Channel and distribution gaps
        
        3. **Operational Excellence Gaps**:
           - Customer service deficiencies
           - Operational efficiency issues
           - Technology infrastructure limitations
           - Process and delivery gaps
        
        4. **Strategic Capability Gaps**:
           - Innovation and R&D weaknesses
           - Partnership and ecosystem gaps
           - Financial or resource constraints
           - Talent and expertise limitations
        
        Return gap analysis and opportunities:
        {{
          "competitive_gaps": [{{
            "gap_type": "service/positioning/operational/strategic",
            "gap_description": "detailed_gap_description",
            "affected_competitors": ["competitor1", "competitor2"],
            "customer_impact": "high/medium/low",
            "opportunity_size": "large/medium/small",
            "exploitation_difficulty": "easy/moderate/difficult",
            "competitive_advantage_potential": "sustainable/temporary/minimal"
          }}],
          "white_space_opportunities": [{{
            "opportunity_name": "opportunity_description",
            "opportunity_type": "market_segment/service_offering/positioning/geographic",
            "description": "detailed_opportunity_description",
            "target_customers": ["customer_segment1", "customer_segment2"],
            "value_proposition": "unique_value_opportunity",
            "market_size_estimate": "large/medium/small",
            "competitive_protection": "high/medium/low",
            "investment_required": "high/medium/low",
            "time_to_market": "fast/moderate/slow",
            "success_probability": 0.0-1.0
          }}],
          "gap_exploitation_strategies": [{{
            "strategy_name": "strategy_description",
            "target_gaps": ["gap1", "gap2"],
            "competitive_response_risk": "high/medium/low",
            "implementation_complexity": "high/medium/low",
            "expected_impact": "transformative/significant/moderate"
          }}],
          "market_entry_opportunities": [{{
            "entry_point": "market_entry_description",
            "entry_barriers": ["barrier1", "barrier2"],
            "competitive_advantages_needed": ["advantage1", "advantage2"],
            "differentiation_requirements": ["requirement1", "requirement2"]
          }}],
          "analysis_confidence": 0.0-1.0
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=gaps_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2000
            )
            
            gaps_result = safe_json_parse(response, default={})
            
            competitive_gaps = gaps_result.get('competitive_gaps', [])
            opportunities = gaps_result.get('white_space_opportunities', [])
            
            if not competitive_gaps and not opportunities:
                competitive_gaps, opportunities = self._get_fallback_gaps_opportunities(competitor_profiles)
            
            # Add analysis metadata
            for gap in competitive_gaps:
                gap['analysis_metadata'] = {
                    'identified_at': time.time(),
                    'gap_id': f"gap_{hash(gap.get('gap_description', ''))}"
                }
            
            for opportunity in opportunities:
                opportunity['analysis_metadata'] = {
                    'identified_at': time.time(),
                    'opportunity_id': f"opp_{hash(opportunity.get('opportunity_name', ''))}"
                }
            
            return competitive_gaps, opportunities
            
        except Exception as e:
            self.logger.error(f"Competitive gap analysis failed: {e}")
            return self._get_fallback_gaps_opportunities(competitor_profiles)

    async def _assess_competitive_threats(
        self, 
        competitor_profiles: List[CompetitorProfile],
        target_market: str,
        service_category: str
    ) -> Dict[str, Any]:
        """
        Assess competitive threats and risks
        
        Args:
            competitor_profiles: List of competitor profiles
            target_market: Target market context
            service_category: Service category context
            
        Returns:
            Comprehensive threat assessment
        """
        
        # Prepare threat analysis data
        high_threat_competitors = [
            p for p in competitor_profiles 
            if p.market_position in ['leader', 'challenger'] or 
               p.growth_trajectory == 'rapid' or
               'strong' in p.financial_health.lower()
        ]
        
        emerging_competitors = [
            p for p in competitor_profiles 
            if p.size_category in ['startup', 'small'] and 
               p.growth_trajectory in ['rapid', 'steady'] and
               p.innovation_capacity in ['high', 'medium']
        ]
        
        threat_prompt = f"""
        Assess competitive threats for {service_category} market in {target_market}:
        
        High Threat Competitors: {len(high_threat_competitors)} identified
        Emerging Competitors: {len(emerging_competitors)} identified
        Total Competitors Analyzed: {len(competitor_profiles)}
        
        Competitor Threat Indicators:
        {json.dumps([{
            'name': p.company_name,
            'position': p.market_position,
            'growth': p.growth_trajectory,
            'advantages': p.competitive_advantages[:2]
        } for p in high_threat_competitors], indent=2)[:2000]}
        
        Assess competitive threats across multiple dimensions:
        
        1. **Direct Competitive Threats**:
           - Head-to-head competition risks
           - Market share erosion threats
           - Pricing pressure risks
           - Customer acquisition competition
        
        2. **Strategic Competitive Threats**:
           - Innovation and technology threats
           - Business model disruption
           - Strategic partnership threats
           - Market expansion threats
        
        3. **Emerging Competitor Threats**:
           - New entrant disruption potential
           - Startup innovation threats
           - Technology-enabled competitors
           - Platform-based competition
        
        4. **Market Disruption Risks**:
           - Industry transformation threats
           - Regulatory change impacts
           - Technology obsolescence risks
           - Customer behavior shift risks
        
        Return comprehensive threat assessment:
        {{
          "immediate_threats": [{{
            "threat_source": "competitor_name_or_trend",
            "threat_type": "direct_competition/pricing/innovation/market_share",
            "threat_description": "detailed_threat_description",
            "threat_level": "critical/high/medium/low",
            "likelihood": "very_likely/likely/possible/unlikely",
            "time_horizon": "immediate/short_term/medium_term/long_term",
            "potential_impact": "severe/significant/moderate/minimal",
            "affected_areas": ["market_share", "pricing", "customers", "revenue"],
            "mitigation_strategies": ["strategy1", "strategy2"],
            "early_warning_signals": ["signal1", "signal2"]
          }}],
          "emerging_threats": [{{
            "threat_source": "emerging_competitor_or_trend",
            "disruption_factor": "innovation/technology/business_model/market_access",
            "threat_description": "potential_disruption_description",
            "development_stage": "early/developing/approaching/imminent",
            "disruption_potential": "transformative/significant/moderate/limited",
            "preparation_time": "months/years",
            "response_complexity": "high/medium/low"
          }}],
          "strategic_risks": [{{
            "risk_category": "innovation/partnership/expansion/regulation",
            "risk_description": "strategic_risk_description",
            "risk_probability": 0.0-1.0,
            "business_impact": "critical/high/medium/low",
            "risk_mitigation": ["mitigation1", "mitigation2"],
            "monitoring_requirements": ["requirement1", "requirement2"]
          }}],
          "competitive_intelligence_priorities": [{{
            "monitoring_target": "competitor_or_trend_to_monitor",
            "monitoring_focus": "what_to_watch_for",
            "monitoring_frequency": "continuous/weekly/monthly/quarterly",
            "intelligence_sources": ["source1", "source2"],
            "success_metrics": ["metric1", "metric2"]
          }}],
          "threat_mitigation_framework": {{
            "defensive_strategies": ["strategy1", "strategy2"],
            "offensive_strategies": ["strategy1", "strategy2"],
            "contingency_plans": ["plan1", "plan2"],
            "competitive_advantages_to_build": ["advantage1", "advantage2"]
          }},
          "overall_threat_level": "critical/high/medium/low",
          "market_stability": "stable/volatile/disrupting",
          "assessment_confidence": 0.0-1.0
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=threat_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2500
            )
            
            threat_assessment = safe_json_parse(response, default={})
            
            if not threat_assessment:
                threat_assessment = self._get_fallback_threat_assessment(competitor_profiles)
            
            # Organize threat assessment results
            organized_assessment = {
                'threats': threat_assessment.get('immediate_threats', []),
                'emerging_competitors': threat_assessment.get('emerging_threats', []),
                'disruption_risks': threat_assessment.get('strategic_risks', []),
                'intelligence_priorities': threat_assessment.get('competitive_intelligence_priorities', []),
                'mitigation_framework': threat_assessment.get('threat_mitigation_framework', {}),
                'overall_assessment': {
                    'threat_level': threat_assessment.get('overall_threat_level', 'medium'),
                    'market_stability': threat_assessment.get('market_stability', 'stable'),
                    'assessment_confidence': threat_assessment.get('assessment_confidence', 0.6)
                }
            }
            
            return organized_assessment
            
        except Exception as e:
            self.logger.error(f"Competitive threat assessment failed: {e}")
            return self._get_fallback_threat_assessment(competitor_profiles)

    async def _generate_strategic_recommendations(
        self, 
        competitor_profiles: List[CompetitorProfile],
        positioning_analysis: Dict[str, Any],
        gap_analysis: List[Dict[str, Any]],
        threat_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate strategic recommendations based on competitive analysis
        
        Args:
            competitor_profiles: Competitor profiles
            positioning_analysis: Positioning analysis results
            gap_analysis: Competitive gap analysis
            threat_assessment: Threat assessment results
            
        Returns:
            Strategic recommendations
        """
        
        recommendations_prompt = f"""
        Generate strategic recommendations based on comprehensive competitive analysis:
        
        Analysis Summary:
        - Competitors Analyzed: {len(competitor_profiles)}
        - Competitive Gaps Identified: {len(gap_analysis)}
        - Threats Detected: {len(threat_assessment.get('threats', []))}
        - Overall Threat Level: {threat_assessment.get('overall_assessment', {}).get('threat_level', 'unknown')}
        
        Key Competitive Insights:
        - Market Positioning: {positioning_analysis.get('positioning_insights', [])}
        - White Space Opportunities: {len(positioning_analysis.get('positioning_map', {}).get('white_spaces', []))}
        - Strategic Groups: {len(positioning_analysis.get('strategic_groups', []))}
        
        Generate strategic recommendations across key areas:
        
        1. **Competitive Positioning Recommendations**:
           - Optimal market positioning strategy
           - Differentiation opportunities and approaches
           - Brand positioning and messaging
           - Value proposition optimization
        
        2. **Competitive Strategy Recommendations**:
           - Market entry and expansion strategies
           - Competitive response strategies
           - Innovation and development priorities
           - Partnership and alliance opportunities
        
        3. **Operational Excellence Recommendations**:
           - Service quality and delivery improvements
           - Customer experience enhancements
           - Operational efficiency opportunities
           - Technology and infrastructure investments
        
        4. **Risk Mitigation Recommendations**:
           - Competitive threat response strategies
           - Market disruption preparation
           - Defensive positioning moves
           - Contingency planning priorities
        
        Return strategic recommendations:
        {{
          "positioning_recommendations": [{{
            "recommendation_type": "positioning/differentiation/branding/value_proposition",
            "recommendation": "specific_strategic_recommendation",
            "rationale": "why_this_recommendation_makes_sense",
            "expected_impact": "transformative/significant/moderate/minimal",
            "implementation_complexity": "high/medium/low",
            "timeline": "immediate/short_term/medium_term/long_term",
            "investment_required": "high/medium/low",
            "competitive_advantage": "sustainable/temporary/none",
            "success_metrics": ["metric1", "metric2"],
            "implementation_steps": ["step1", "step2"]
          }}],
          "competitive_strategies": [{{
            "strategy_name": "strategy_description",
            "strategic_objective": "what_this_strategy_achieves",
            "target_competitors": ["competitor1", "competitor2"],
            "competitive_moves": ["move1", "move2"],
            "success_factors": ["factor1", "factor2"],
            "risk_factors": ["risk1", "risk2"],
            "resource_requirements": ["requirement1", "requirement2"]
          }}],
          "operational_recommendations": [{{
            "operational_area": "service_delivery/customer_experience/technology/efficiency",
            "improvement_opportunity": "specific_improvement_description",
            "competitive_benefit": "how_this_creates_competitive_advantage",
            "implementation_priority": "critical/high/medium/low",
            "quick_wins": ["quick_win1", "quick_win2"],
            "long_term_investments": ["investment1", "investment2"]
          }}],
          "risk_mitigation_strategies": [{{
            "risk_category": "competitive/market/technology/operational",
            "mitigation_strategy": "specific_mitigation_approach",
            "preventive_measures": ["measure1", "measure2"],
            "response_protocols": ["protocol1", "protocol2"],
            "monitoring_indicators": ["indicator1", "indicator2"]
          }}],
          "competitive_monitoring_priorities": [
            "competitor_or_trend_to_monitor_closely",
            "key_market_indicator_to_track",
            "emerging_threat_to_watch"
          ],
          "strategic_priorities": {{
            "immediate_actions": ["action1", "action2"],
            "short_term_initiatives": ["initiative1", "initiative2"],
            "long_term_investments": ["investment1", "investment2"],
            "continuous_monitoring": ["monitoring1", "monitoring2"]
          }},
          "recommendations_confidence": 0.0-1.0
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=recommendations_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2500
            )
            
            recommendations = safe_json_parse(response, default={})
            
            if not recommendations:
                recommendations = self._get_fallback_recommendations()
            
            # Organize recommendations
            organized_recommendations = {
                'positioning': recommendations.get('positioning_recommendations', []),
                'strategies': recommendations.get('competitive_strategies', []),
                'operational': recommendations.get('operational_recommendations', []),
                'risk_mitigation': recommendations.get('risk_mitigation_strategies', []),
                'monitoring_priorities': recommendations.get('competitive_monitoring_priorities', []),
                'strategic_priorities': recommendations.get('strategic_priorities', {}),
                'recommendations_metadata': {
                    'generated_at': time.time(),
                    'confidence': recommendations.get('recommendations_confidence', 0.6),
                    'based_on_competitors': len(competitor_profiles)
                }
            }
            
            return organized_recommendations
            
        except Exception as e:
            self.logger.error(f"Strategic recommendations generation failed: {e}")
            return self._get_fallback_recommendations()

    # Utility and helper methods
    
    def _generate_competitor_id(self, competitor_name: str) -> str:
        """Generate unique competitor ID"""
        return f"comp_{hashlib.md5(competitor_name.encode()).hexdigest()[:8]}"

    def _summarize_competitive_landscape(self, competitor_profiles: List[CompetitorProfile]) -> Dict[str, Any]:
        """Summarize competitive landscape from competitor profiles"""
        if not competitor_profiles:
            return {'total_competitors': 0, 'landscape_summary': 'No competitors analyzed'}
        
        # Analyze competitor distribution
        position_distribution = {}
        size_distribution = {}
        
        for profile in competitor_profiles:
            position = profile.market_position
            size = profile.size_category
            
            position_distribution[position] = position_distribution.get(position, 0) + 1
            size_distribution[size] = size_distribution.get(size, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(p.analysis_confidence for p in competitor_profiles) / len(competitor_profiles)
        
        return {
            'total_competitors': len(competitor_profiles),
            'position_distribution': position_distribution,
            'size_distribution': size_distribution,
            'average_analysis_confidence': avg_confidence,
            'competitive_intensity': 'high' if len(competitor_profiles) > 10 else 'medium' if len(competitor_profiles) > 5 else 'low'
        }

    async def _assess_analysis_quality(
        self, 
        competitor_profiles: List[CompetitorProfile],
        positioning_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess quality of competitive analysis"""
        
        # Calculate data completeness
        total_profiles = len(competitor_profiles)
        high_confidence_profiles = len([p for p in competitor_profiles if p.analysis_confidence > 0.7])
        
        data_completeness = high_confidence_profiles / max(1, total_profiles)
        
        # Calculate overall confidence
        profile_confidences = [p.analysis_confidence for p in competitor_profiles]
        positioning_confidence = positioning_analysis.get('analysis_confidence', 0.5)
        
        overall_confidence = (
            sum(profile_confidences) / max(1, len(profile_confidences)) * 0.7 +
            positioning_confidence * 0.3
        )
        
        return {
            'overall_confidence': overall_confidence,
            'data_completeness': data_completeness,
            'high_confidence_profiles': high_confidence_profiles,
            'total_profiles_analyzed': total_profiles,
            'quality_indicators': [
                'ai_powered_analysis',
                'multi_dimensional_assessment',
                'strategic_framework_based'
            ],
            'analysis_limitations': [
                'ai_estimation_based',
                'limited_real_time_data',
                'competitive_intelligence_approximation'
            ]
        }

    def _update_analysis_time(self, analysis_duration: float):
        """Update average analysis time tracking"""
        current_count = self.competitive_stats['total_analyses']
        if current_count == 1:
            self.competitive_stats['average_analysis_time'] = analysis_duration
        else:
            current_avg = self.competitive_stats['average_analysis_time']
            self.competitive_stats['average_analysis_time'] = (
                (current_avg * (current_count - 1) + analysis_duration) / current_count
            )

    # Fallback methods for AI failure scenarios
    
    def _get_fallback_positioning_analysis(self, competitor_profiles: List[CompetitorProfile]) -> Dict[str, Any]:
        """Provide fallback positioning analysis"""
        return {
            'positioning_dimensions': {
                'primary_dimensions': [{'dimension_name': 'price_value', 'competitive_spread': 'dispersed'}],
                'positioning_insights': ['competitive landscape analysis incomplete']
            },
            'strategic_groups': [{'group_name': 'general competitors', 'members': [p.company_name for p in competitor_profiles[:3]]}],
            'analysis_confidence': 0.3
        }

    def _get_fallback_gaps_opportunities(self, competitor_profiles: List[CompetitorProfile]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Provide fallback gaps and opportunities"""
        gaps = [{'gap_type': 'analysis_limitation', 'gap_description': 'Detailed gap analysis unavailable', 'opportunity_size': 'unknown'}]
        opportunities = [{'opportunity_name': 'Market Analysis', 'opportunity_type': 'intelligence', 'description': 'Requires detailed competitive analysis'}]
        return gaps, opportunities

    def _get_fallback_threat_assessment(self, competitor_profiles: List[CompetitorProfile]) -> Dict[str, Any]:
        """Provide fallback threat assessment"""
        return {
            'threats': [{'threat_source': 'general_competition', 'threat_level': 'medium', 'threat_description': 'Standard competitive pressure'}],
            'emerging_competitors': [],
            'disruption_risks': [],
            'overall_assessment': {'threat_level': 'medium', 'assessment_confidence': 0.3}
        }

    def _get_fallback_recommendations(self) -> Dict[str, Any]:
        """Provide fallback recommendations"""
        return {
            'positioning': [{'recommendation': 'Conduct detailed competitive analysis', 'implementation_complexity': 'medium'}],
            'strategies': [{'strategy_name': 'Market Research', 'strategic_objective': 'Better competitive understanding'}],
            'monitoring_priorities': ['competitor_activity', 'market_changes'],
            'recommendations_metadata': {'confidence': 0.3}
        }

    # Agent lifecycle methods
    
    async def _test_competitive_analysis_capabilities(self):
        """Test AI capabilities for competitive analysis"""
        test_prompt = "Test competitive analysis capability. What are key factors for competitor analysis? Return JSON format."
        
        try:
            response = await self.ask_ai(
                prompt=test_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.SIMPLE
            )
            
            result = safe_json_parse(response, default={})
            if result and ('competitive' in str(result).lower() or 'competitor' in str(result).lower()):
                self.logger.debug("Competitive analysis capability test: SUCCESS")
            else:
                self.logger.warning("Competitive analysis capability test: PARTIAL")
                
        except Exception as e:
            self.logger.warning(f"Competitive analysis capability test failed: {e}")

    async def _initialize_competitive_databases(self):
        """Initialize competitive intelligence databases"""
        # Placeholder for competitive intelligence database initialization
        self.logger.debug("Competitive intelligence databases initialized")

    async def _setup_monitoring_frameworks(self):
        """Setup competitive monitoring frameworks"""
        # Placeholder for monitoring framework setup
        self.logger.debug("Competitive monitoring frameworks initialized")

    async def _save_competitive_intelligence(self):
        """Save competitive intelligence data"""
        try:
            intelligence_summary = {
                'total_analyses': self.competitive_stats['total_analyses'],
                'competitors_identified': self.competitive_stats['competitors_identified'],
                'threats_detected': self.competitive_stats['threats_detected'],
                'opportunities_found': self.competitive_stats['opportunities_found']
            }
            
            self.logger.info(f"Competitive intelligence summary: {intelligence_summary}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save competitive intelligence: {e}")

    async def _export_competitive_metrics(self):
        """Export competitive analysis metrics"""
        # Placeholder for metrics export functionality
        self.logger.debug("Competitive analysis metrics exported")

    def get_competitive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive competitive analysis statistics"""
        return {
            'performance_metrics': self.competitive_stats,
            'configuration': self.analysis_config,
            'frameworks_available': list(self.analysis_frameworks.keys())
        }