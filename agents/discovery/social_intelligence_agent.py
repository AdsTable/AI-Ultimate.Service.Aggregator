# agents/discovery/social_intelligence_agent.py - not complete!
"""
Social Intelligence Agent

This agent specializes in discovering service providers through social media
analysis, online community monitoring, and digital presence intelligence. It uses
AI to understand social signals and identify providers through their digital footprint.
"""

import asyncio
import logging
import json
import time
import re
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse, quote_plus
import aiohttp
from bs4 import BeautifulSoup
import hashlib

from ..base import BaseAgent, AgentConfig, AgentError, safe_json_parse, retry_with_backoff, rate_limiter
from .models import DiscoveryTarget, ProviderCandidate, SearchStrategy
from services.ai_async_client import AIAsyncClient, TaskComplexity

logger = logging.getLogger(__name__)


class SocialIntelligenceAgent(BaseAgent):
    """
    Specialized agent for social media and digital presence intelligence
    
    Features:
    - Multi-platform social media analysis
    - Professional network intelligence (LinkedIn focus)
    - Community and forum monitoring
    - Social sentiment analysis
    - Digital presence mapping
    - Influencer and thought leader identification
    
    Note: This implementation respects platform rate limits and terms of service.
    For production use, consider implementing official API integrations where available.
    """
    
    def __init__(self, ai_client: AIAsyncClient, redis_client=None):
        config = AgentConfig(
            name="SocialIntelligenceAgent",
            max_retries=2,
            rate_limit=20,  # Conservative for social platforms
            preferred_ai_provider="ollama",  # Use free provider for social analysis
            task_complexity=TaskComplexity.MEDIUM,
            cache_ttl=1800,  # 30 minutes cache for social data
            debug=False
        )
        
        super().__init__(config, ai_client)
        
        # Redis client for caching social intelligence
        self.redis_client = redis_client
        
        # HTTP session for social media research
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Social media platforms and their characteristics
        self.social_platforms = {
            'linkedin': {
                'base_url': 'https://www.linkedin.com',
                'search_endpoint': '/search/results/companies/',
                'rate_limit': 5.0,  # LinkedIn is very strict
                'data_format': 'html',
                'business_focused': True,
                'requires_auth': True,
                'priority': 10  # Highest priority for business discovery
            },
            'twitter': {
                'base_url': 'https://twitter.com',
                'search_endpoint': '/search',
                'rate_limit': 3.0,
                'data_format': 'html',
                'business_focused': False,
                'requires_auth': False,
                'priority': 7
            },
            'facebook': {
                'base_url': 'https://www.facebook.com',
                'search_endpoint': '/search/pages',
                'rate_limit': 4.0,
                'data_format': 'html',
                'business_focused': False,
                'requires_auth': False,
                'priority': 6
            },
            'youtube': {
                'base_url': 'https://www.youtube.com',
                'search_endpoint': '/results',
                'rate_limit': 2.0,
                'data_format': 'html',
                'business_focused': False,
                'requires_auth': False,
                'priority': 5
            },
            'reddit': {
                'base_url': 'https://www.reddit.com',
                'search_endpoint': '/search',
                'rate_limit': 3.0,
                'data_format': 'html',
                'business_focused': False,
                'requires_auth': False,
                'priority': 8  # High for community insights
            }
        }
        
        # Industry-specific communities and forums
        self.industry_communities = {
            'technology': [
                'hacker_news', 'stack_overflow', 'reddit_programming',
                'dev_community', 'product_hunt', 'tech_crunch'
            ],
            'business': [
                'reddit_entrepreneur', 'indie_hackers', 'hacker_news',
                'medium_business', 'quora_business', 'startup_forums'
            ],
            'finance': [
                'reddit_finance', 'seeking_alpha', 'investopedia',
                'fintech_forums', 'bloomberg_terminal', 'financial_planning'
            ],
            'healthcare': [
                'reddit_medicine', 'healthcare_forums', 'med_tech_communities',
                'digital_health_forums', 'telemedicine_groups'
            ],
            'education': [
                'edtech_forums', 'reddit_education', 'teaching_communities',
                'e_learning_groups', 'academic_twitter'
            ]
        }
        
        # Social intelligence cache
        self.social_cache = {}
        
        # Performance tracking
        self.intelligence_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'platforms_analyzed': 0,
            'providers_discovered': 0,
            'community_mentions': 0,
            'platform_performance': {}
        }
    
    async def _setup_agent(self) -> None:
        """Initialize HTTP session and social platform configurations"""
        try:
            # Setup HTTP session with social media appropriate settings
            timeout = aiohttp.ClientTimeout(total=25)
            connector = aiohttp.TCPConnector(
                limit=6,  # Conservative for social platforms
                limit_per_host=3,
                ttl_dns_cache=300
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Research Bot/1.0 (Academic Business Research)',
                    'Accept': 'text/html,application/json,*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1'  # Do Not Track header for privacy
                }
            )
            
            # Load social intelligence cache
            await self._load_social_cache()
            
            self.logger.info("Social intelligence agent initialized successfully")
            
        except Exception as e:
            raise AgentError(self.config.name, f"Failed to initialize social intelligence agent: {e}")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup HTTP session and save cache"""
        try:
            # Save social intelligence cache
            await self._save_social_cache()
            
            # Close HTTP session
            if self.session and not self.session.closed:
                await self.session.close()
                await asyncio.sleep(0.1)
            
            self.logger.info("Social intelligence agent cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup warning for social intelligence agent: {e}")
    
    async def gather_social_intelligence(
        self, 
        target: DiscoveryTarget,
        strategies: List[SearchStrategy]
    ) -> List[ProviderCandidate]:
        """
        Main social intelligence gathering method for discovering providers
        
        Args:
            target: Discovery target specification
            strategies: Social intelligence strategies
            
        Returns:
            List of providers discovered through social intelligence
        """
        self.logger.info(f"📱 Starting social intelligence gathering for {target.service_category} in {target.country}")
        
        start_time = time.time()
        all_candidates = []
        
        try:
            # Filter for social intelligence strategies
            social_strategies = [s for s in strategies if s.method == 'social_intelligence']
            
            if not social_strategies:
                # Generate default social intelligence strategy
                social_strategies = await self._generate_default_social_strategy(target)
            
            # Phase 1: Analyze social landscape for target market
            social_landscape = await self._analyze_social_landscape(target)
            
            # Phase 2: Execute social intelligence strategies
            for strategy in social_strategies:
                strategy_candidates = await self._execute_social_strategy(
                    target, strategy, social_landscape
                )
                all_candidates.extend(strategy_candidates)
                
                # Rate limiting between strategies
                await asyncio.sleep(2.0)
            
            # Phase 3: Multi-platform social discovery
            platform_candidates = await self._multi_platform_discovery(target, social_landscape)
            all_candidates.extend(platform_candidates)
            
            # Phase 4: Community and forum analysis
            community_candidates = await self._analyze_industry_communities(target, social_landscape)
            all_candidates.extend(community_candidates)
            
            # Phase 5: Validate and enhance candidates with social metrics
            validated_candidates = await self._validate_social_candidates(
                all_candidates, target, social_landscape
            )
            
            intelligence_time = time.time() - start_time
            self.intelligence_stats['total_searches'] += 1
            if validated_candidates:
                self.intelligence_stats['successful_searches'] += 1
                self.intelligence_stats['providers_discovered'] += len(validated_candidates)
            
            self.logger.info(
                f"✅ Social intelligence gathering complete: {len(validated_candidates)} candidates "
                f"found in {intelligence_time:.1f}s"
            )
            
            return validated_candidates
            
        except Exception as e:
            self.logger.error(f"❌ Social intelligence gathering failed: {e}")
            raise AgentError(self.config.name, f"Social intelligence gathering failed: {e}")
    
    async def _analyze_social_landscape(self, target: DiscoveryTarget) -> Dict[str, Any]:
        """
        Use AI to analyze the social media landscape for the target market
        
        Args:
            target: Discovery target with market context
            
        Returns:
            Social landscape analysis with platform insights and strategies
        """
        
        landscape_prompt = f"""
        Analyze the social media landscape for {target.service_category} businesses in {target.country}.
        
        Target Market Context:
        - Country: {target.country}
        - Service Category: {target.service_category}
        - Language: {target.language}
        - Market Size: {target.market_size_estimate}
        
        Analyze social media presence patterns for this industry:
        
        1. **Platform Preferences**: Which social platforms are most used by {target.service_category} providers
        2. **Content Patterns**: What type of content these businesses typically share
        3. **Community Hubs**: Key online communities and forums for this industry
        4. **Professional Networks**: How businesses in this sector use LinkedIn and professional platforms
        5. **Customer Interaction**: How customers discuss and review these services online
        6. **Influencer Landscape**: Key thought leaders and influencers in this space
        
        Consider regional variations for {target.country}:
        - Local social platforms popular in {target.country}
        - Language-specific communities in {target.language}
        - Cultural preferences for business communication
        - Regional professional networks
        
        Return JSON with social landscape analysis:
        {{
          "platform_priorities": [{{
            "platform": "platform_name",
            "business_usage": "high/medium/low",
            "discovery_potential": "high/medium/low",
            "content_types": ["company_pages", "executive_profiles", "case_studies"],
            "search_strategies": ["hashtags", "keywords", "location_tags"]
          }}],
          "industry_communities": [{{
            "community_name": "forum_or_group_name",
            "platform": "where_it_exists",
            "member_count": "estimated_size",
            "relevance": "high/medium/low",
            "access_level": "public/private/restricted"
          }}],
          "social_signals": [{{
            "signal_type": "mentions/reviews/partnerships",
            "platforms": ["where_to_find"],
            "search_patterns": ["how_to_search"],
            "reliability": "high/medium/low"
          }}],
          "content_strategies": {{
            "hashtag_patterns": ["relevant_hashtags"],
            "keyword_variations": ["search_terms"],
            "geographic_modifiers": ["location_tags"],
            "industry_terms": ["sector_specific_terms"]
          }},
          "discovery_challenges": ["platform_limitations", "privacy_restrictions"],
          "recommended_approach": "prioritized_discovery_strategy"
        }}
        """
        
        try:
            response = await self.ask_ai(
                prompt=landscape_prompt,
                provider="ollama",  # Use free provider for landscape analysis
                task_complexity=TaskComplexity.COMPLEX,
                max_tokens=2000
            )
            
            landscape_data = safe_json_parse(response, default={})
            
            if not landscape_data:
                self.logger.warning("AI returned empty social landscape, using fallback")
                return self._get_fallback_social_landscape(target)
            
            self.logger.info(f"📊 Social landscape analyzed: {len(landscape_data.get('platform_priorities', []))} platforms prioritized")
            return landscape_data
            
        except Exception as e:
            self.logger.error(f"Social landscape analysis failed: {e}")
            return self._get_fallback_social_landscape(target)
    
    async def _execute_social_strategy(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        social_landscape: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Execute a specific social intelligence strategy
        
        Args:
            target: Discovery target
            strategy: Social intelligence strategy
            social_landscape: Social landscape analysis
            
        Returns:
            List of candidates from social strategy execution
        """
        self.logger.info(f"🎯 Executing social strategy: {strategy.method}")
        
        candidates = []
        
        # Get platform priorities from landscape analysis
        platform_priorities = social_landscape.get('platform_priorities', [])
        
        # Execute strategy across prioritized platforms
        for platform_info in platform_priorities:
            platform_name = platform_info.get('platform', '')
            discovery_potential = platform_info.get('discovery_potential', 'low')
            
            # Skip low-potential platforms unless specifically requested
            if discovery_potential == 'low' and platform_name not in strategy.platforms:
                continue
            
            try:
                platform_candidates = await self._search_social_platform(
                    platform_name, target, strategy, platform_info
                )
                candidates.extend(platform_candidates)
                
                # Track platform performance
                if platform_name not in self.intelligence_stats['platform_performance']:
                    self.intelligence_stats['platform_performance'][platform_name] = {
                        'searches': 0,
                        'candidates_found': 0
                    }
                
                platform_stats = self.intelligence_stats['platform_performance'][platform_name]
                platform_stats['searches'] += 1
                platform_stats['candidates_found'] += len(platform_candidates)
                
                # Respectful rate limiting
                platform_config = self.social_platforms.get(platform_name, {})
                rate_limit = platform_config.get('rate_limit', 3.0)
                await asyncio.sleep(rate_limit)
                
            except Exception as e:
                self.logger.error(f"Social platform search failed for {platform_name}: {e}")
                continue
        
        self.intelligence_stats['platforms_analyzed'] += len(platform_priorities)
        
        self.logger.info(f"📋 Social strategy executed: {len(candidates)} candidates found")
        return candidates
    
    @rate_limiter(max_calls=30, time_window=300)  # 30 calls per 5 minutes
    async def _search_social_platform(
        self, 
        platform_name: str,
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        platform_info: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Search a specific social platform for provider candidates
        
        Args:
            platform_name: Name of social platform
            target: Discovery target
            strategy: Search strategy
            platform_info: Platform-specific information
            
        Returns:
            List of candidates from platform search
        """
        platform_config = self.social_platforms.get(platform_name, {})
        
        if not platform_config:
            self.logger.warning(f"Unknown social platform: {platform_name}")
            return []
        
        # Check cache first
        cache_key = f"social_{platform_name}_{target.country}_{target.service_category}"
        cached_results = await self._get_cached_social_data(cache_key)
        if cached_results:
            self.logger.debug(f"📚 Using cached results for {platform_name}")
            return cached_results
        
        try:
            if platform_name == 'linkedin':
                candidates = await self._search_linkedin(target, strategy, platform_info)
            elif platform_name == 'twitter':
                candidates = await self._search_twitter(target, strategy, platform_info)
            elif platform_name == 'reddit':
                candidates = await self._search_reddit(target, strategy, platform_info)
            elif platform_name == 'facebook':
                candidates = await self._search_facebook(target, strategy, platform_info)
            elif platform_name == 'youtube':
                candidates = await self._search_youtube(target, strategy, platform_info)
            else:
                # Use AI knowledge for unknown platforms
                candidates = await self._search_platform_via_ai(platform_name, target, strategy)
            
            # Cache successful results
            if candidates:
                await self._cache_social_data(cache_key, candidates)
            
            self.logger.info(f"🔍 {platform_name}: {len(candidates)} candidates found")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Platform search failed for {platform_name}: {e}")
            return []
    
    async def _search_linkedin(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        platform_info: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Search LinkedIn for business providers (AI knowledge-based due to LinkedIn's restrictions)
        
        Args:
            target: Discovery target
            strategy: Search strategy
            platform_info: LinkedIn-specific information
            
        Returns:
            List of candidates from LinkedIn analysis
        """
        
        # LinkedIn has strict anti-scraping policies, so we use AI knowledge
        linkedin_prompt = f"""
        Based on your knowledge of LinkedIn business presence, identify {target.service_category} providers in {target.country}.
        
        LinkedIn Business Analysis:
        - Service Category: {target.service_category}
        - Country: {target.country}
        - Language: {target.language}
        
        Focus on companies that would typically have strong LinkedIn presence:
        1. **B2B Service Providers**: Companies targeting business customers
        2. **Professional Services**: Consulting, advisory, technical services
        3. **Technology Companies**: Software, SaaS, IT services
        4. **Enterprise Solutions**: Large-scale service providers
        5. **Regional Leaders**: Companies with significant local presence
        
        Identify providers based on LinkedIn activity patterns:
        - Companies with active company pages
        - Businesses with executive thought leadership
        - Organizations posting industry content
        - Companies with employee advocacy programs
        - Businesses participating in professional groups
        
        Return JSON with LinkedIn-discovered providers:
        [{{
          "company_name": "business_name",
          "estimated_website": "company_website",
          "linkedin_presence": "strong/medium/emerging",
          "business_type": "b2b/b2c/enterprise",
          "employee_count": "estimated_size",
          "industry_focus": ["specific_industry_sectors"],
          "linkedin_activity": ["content_posting", "thought_leadership", "employee_advocacy"],
          "professional_credibility": ["certifications", "partnerships", "awards"],
          "discovery_reasoning": "why_this_company_has_strong_linkedin_presence"
        }}]
        
        Focus on 8-10 companies with strongest professional presence indicators.
        """
        
        try:
            response = await self.ask_ai(
                prompt=linkedin_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1200
            )
            
            linkedin_data = safe_json_parse(response, default=[])
            
            candidates = []
            for company_data in linkedin_data:
                try:
                    # Calculate confidence based on LinkedIn presence indicators
                    linkedin_presence = company_data.get('linkedin_presence', '').lower()
                    business_type = company_data.get('business_type', '')
                    linkedin_activity = company_data.get('linkedin_activity', [])
                    
                    base_confidence = 0.6  # Higher base for LinkedIn (professional platform)
                    
                    # Boost confidence for strong presence
                    if linkedin_presence == 'strong':
                        base_confidence += 0.2
                    elif linkedin_presence == 'medium':
                        base_confidence += 0.1
                    
                    # Boost for B2B companies (more likely to use LinkedIn actively)
                    if 'b2b' in business_type or 'enterprise' in business_type:
                        base_confidence += 0.1
                    
                    # Boost for active LinkedIn engagement
                    activity_boost = min(0.1, len(linkedin_activity) * 0.03)
                    final_confidence = min(0.85, base_confidence + activity_boost)
                    
                    candidate = ProviderCandidate(
                        name=company_data.get('company_name', 'Unknown'),
                        website=company_data.get('estimated_website', ''),
                        discovery_method='social_linkedin_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='professional_network',
                        ai_analysis={
                            'source': 'linkedin_analysis',
                            'linkedin_presence': linkedin_presence,
                            'business_type': business_type,
                            'employee_count': company_data.get('employee_count', 'unknown'),
                            'industry_focus': company_data.get('industry_focus', []),
                            'linkedin_activity': linkedin_activity,
                            'professional_credibility': company_data.get('professional_credibility', []),
                            'discovery_reasoning': company_data.get('discovery_reasoning', '')
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create LinkedIn candidate: {e}")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"LinkedIn analysis failed: {e}")
            return []
    
    async def _search_twitter(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        platform_info: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Search Twitter for provider mentions and discussions
        
        Args:
            target: Discovery target
            strategy: Search strategy
            platform_info: Twitter-specific information
            
        Returns:
            List of candidates from Twitter analysis
        """
        
        twitter_prompt = f"""
        Analyze Twitter discussions and mentions to discover {target.service_category} providers in {target.country}.
        
        Twitter Analysis Context:
        - Service Category: {target.service_category}
        - Country: {target.country}
        - Language: {target.language}
        
        Identify providers through Twitter activity:
        1. **Active Business Accounts**: Companies with regular Twitter presence
        2. **Customer Mentions**: Businesses mentioned in customer tweets
        3. **Industry Discussions**: Companies participating in sector conversations
        4. **News and Announcements**: Businesses mentioned in industry news
        5. **Influencer Recommendations**: Companies endorsed by industry influencers
        
        Twitter discovery patterns:
        - Companies with verified business accounts
        - Businesses with strong customer engagement
        - Organizations mentioned in industry hashtags
        - Companies with thought leadership content
        - Businesses referenced in customer support conversations
        
        Return JSON with Twitter-discovered providers:
        [{{
          "company_name": "business_name",
          "twitter_handle": "@company_handle",
          "estimated_website": "company_website",
          "twitter_activity": "high/medium/low",
          "follower_engagement": "strong/moderate/weak",
          "content_focus": ["customer_service", "thought_leadership", "product_updates"],
          "mention_context": ["customer_praise", "industry_discussion", "news_coverage"],
          "influence_indicators": ["verified_account", "industry_recognition", "media_mentions"],
          "discovery_source": "direct_account/customer_mention/industry_discussion"
        }}]
        
        Focus on 6-8 companies with strongest Twitter presence and relevance.
        """
        
        try:
            response = await self.ask_ai(
                prompt=twitter_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1000
            )
            
            twitter_data = safe_json_parse(response, default=[])
            
            candidates = []
            for twitter_info in twitter_data:
                try:
                    # Calculate confidence based on Twitter engagement
                    twitter_activity = twitter_info.get('twitter_activity', '').lower()
                    follower_engagement = twitter_info.get('follower_engagement', '').lower()
                    influence_indicators = twitter_info.get('influence_indicators', [])
                    
                    base_confidence = 0.4  # Lower base for Twitter (less business-focused)
                    
                    # Boost confidence for high activity
                    if twitter_activity == 'high':
                        base_confidence += 0.15
                    elif twitter_activity == 'medium':
                        base_confidence += 0.1
                    
                    # Boost for strong engagement
                    if follower_engagement == 'strong':
                        base_confidence += 0.15
                    elif follower_engagement == 'moderate':
                        base_confidence += 0.1
                    
                    # Boost for influence indicators
                    influence_boost = min(0.2, len(influence_indicators) * 0.07)
                    final_confidence = min(0.75, base_confidence + influence_boost)
                    
                    candidate = ProviderCandidate(
                        name=twitter_info.get('company_name', 'Unknown'),
                        website=twitter_info.get('estimated_website', ''),
                        discovery_method='social_twitter_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='social_active',
                        ai_analysis={
                            'source': 'twitter_analysis',
                            'twitter_handle': twitter_info.get('twitter_handle', ''),
                            'twitter_activity': twitter_activity,
                            'follower_engagement': follower_engagement,
                            'content_focus': twitter_info.get('content_focus', []),
                            'mention_context': twitter_info.get('mention_context', []),
                            'influence_indicators': influence_indicators,
                            'discovery_source': twitter_info.get('discovery_source', '')
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create Twitter candidate: {e}")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Twitter analysis failed: {e}")
            return []
    
    async def _search_reddit(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        platform_info: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Search Reddit for community discussions about providers
        
        Args:
            target: Discovery target
            strategy: Search strategy
            platform_info: Reddit-specific information
            
        Returns:
            List of candidates from Reddit analysis
        """
        
        reddit_prompt = f"""
        Analyze Reddit community discussions to discover {target.service_category} providers in {target.country}.
        
        Reddit Analysis Context:
        - Service Category: {target.service_category}
        - Country: {target.country}
        - Language: {target.language}
        
        Reddit discovery approach:
        1. **Recommendation Threads**: Users asking for and recommending providers
        2. **Review Discussions**: Detailed user experiences with services
        3. **Industry Subreddits**: Professional communities discussing providers
        4. **Local Subreddits**: Regional discussions about local services
        5. **Problem-Solution Threads**: Users sharing solutions and providers
        
        Look for providers mentioned in:
        - "Who should I use for..." threads
        - Service comparison discussions
        - User experience sharing
        - Local business recommendations
        - Industry-specific subreddit discussions
        
        Return JSON with Reddit-discovered providers:
        [{{
          "company_name": "provider_name",
          "estimated_website": "website_url",
          "reddit_mentions": "frequent/occasional/rare",
          "mention_context": ["recommendations", "reviews", "discussions"],
          "user_sentiment": "positive/mixed/negative",
          "discussion_subreddits": ["relevant_subreddits"],
          "credibility_indicators": ["verified_user_posts", "detailed_reviews", "multiple_mentions"],
          "service_feedback": ["user_reported_strengths", "common_complaints"],
          "discovery_confidence": "high/medium/low"
        }}]
        
        Focus on 5-7 providers with strongest Reddit community validation.
        """
        
        try:
            response = await self.ask_ai(
                prompt=reddit_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1000
            )
            
            reddit_data = safe_json_parse(response, default=[])
            
            candidates = []
            for reddit_info in reddit_data:
                try:
                    # Calculate confidence based on Reddit community validation
                    reddit_mentions = reddit_info.get('reddit_mentions', '').lower()
                    user_sentiment = reddit_info.get('user_sentiment', '').lower()
                    credibility_indicators = reddit_info.get('credibility_indicators', [])
                    discovery_confidence = reddit_info.get('discovery_confidence', '').lower()
                    
                    base_confidence = 0.5  # Medium base for Reddit (community validation)
                    
                    # Boost confidence for frequent mentions
                    if reddit_mentions == 'frequent':
                        base_confidence += 0.2
                    elif reddit_mentions == 'occasional':
                        base_confidence += 0.1
                    
                    # Boost for positive sentiment
                    if user_sentiment == 'positive':
                        base_confidence += 0.15
                    elif user_sentiment == 'mixed':
                        base_confidence += 0.05
                    
                    # Boost for credibility indicators
                    credibility_boost = min(0.15, len(credibility_indicators) * 0.05)
                    
                    # Apply discovery confidence
                    if discovery_confidence == 'high':
                        base_confidence += 0.1
                    elif discovery_confidence == 'low':
                        base_confidence -= 0.1
                    
                    final_confidence = min(0.8, max(0.2, base_confidence + credibility_boost))
                    
                    candidate = ProviderCandidate(
                        name=reddit_info.get('company_name', 'Unknown'),
                        website=reddit_info.get('estimated_website', ''),
                        discovery_method='social_reddit_community_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='community_validated',
                        ai_analysis={
                            'source': 'reddit_analysis',
                            'reddit_mentions': reddit_mentions,
                            'mention_context': reddit_info.get('mention_context', []),
                            'user_sentiment': user_sentiment,
                            'discussion_subreddits': reddit_info.get('discussion_subreddits', []),
                            'credibility_indicators': credibility_indicators,
                            'service_feedback': reddit_info.get('service_feedback', []),
                            'discovery_confidence': discovery_confidence
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create Reddit candidate: {e}")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Reddit analysis failed: {e}")
            return []
    
    async def _search_facebook(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        platform_info: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Search Facebook for business pages and community mentions
        
        Args:
            target: Discovery target
            strategy: Search strategy
            platform_info: Facebook-specific information
            
        Returns:
            List of candidates from Facebook analysis
        """
        
        facebook_prompt = f"""
        Analyze Facebook business presence for {target.service_category} providers in {target.country}.
        
        Facebook Business Analysis:
        - Service Category: {target.service_category}
        - Country: {target.country}
        - Language: {target.language}
        
        Facebook discovery focus:
        1. **Business Pages**: Official company Facebook pages
        2. **Local Business Listings**: Businesses with location information
        3. **Customer Reviews**: Facebook reviews and ratings
        4. **Community Groups**: Local business and industry groups
        5. **Event Participation**: Businesses participating in or hosting events
        
        Look for providers with:
        - Verified business pages
        - Active customer engagement
        - Positive review patterns
        - Local community presence
        - Regular content posting
        
        Return JSON with Facebook-discovered providers:
        [{{
          "company_name": "business_name",
          "estimated_website": "website_url",
          "facebook_presence": "verified/unverified",
          "page_activity": "high/medium/low",
          "customer_reviews": "positive/mixed/negative",
          "review_count": "estimated_number",
          "local_presence": "strong/moderate/weak",
          "engagement_indicators": ["regular_posts", "customer_responses", "event_hosting"],
          "business_information": ["contact_details", "location", "hours", "services"]
        }}]
        
        Focus on 6-8 businesses with strongest Facebook business presence.
        """
        
        try:
            response = await self.ask_ai(
                prompt=facebook_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1000
            )
            
            facebook_data = safe_json_parse(response, default=[])
            
            candidates = []
            for fb_info in facebook_data:
                try:
                    # Calculate confidence based on Facebook business indicators
                    facebook_presence = fb_info.get('facebook_presence', '').lower()
                    page_activity = fb_info.get('page_activity', '').lower()
                    customer_reviews = fb_info.get('customer_reviews', '').lower()
                    local_presence = fb_info.get('local_presence', '').lower()
                    
                    base_confidence = 0.4  # Medium-low base for Facebook
                    
                    # Boost confidence for verified presence
                    if facebook_presence == 'verified':
                        base_confidence += 0.15
                    
                    # Boost for high activity
                    if page_activity == 'high':
                        base_confidence += 0.15
                    elif page_activity == 'medium':
                        base_confidence += 0.1
                    
                    # Boost for positive reviews
                    if customer_reviews == 'positive':
                        base_confidence += 0.15
                    elif customer_reviews == 'mixed':
                        base_confidence += 0.05
                    
                    # Boost for strong local presence
                    if local_presence == 'strong':
                        base_confidence += 0.1
                    elif local_presence == 'moderate':
                        base_confidence += 0.05
                    
                    final_confidence = min(0.75, base_confidence)
                    
                    candidate = ProviderCandidate(
                        name=fb_info.get('company_name', 'Unknown'),
                        website=fb_info.get('estimated_website', ''),
                        discovery_method='social_facebook_business_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='local_social_presence',
                        ai_analysis={
                            'source': 'facebook_analysis',
                            'facebook_presence': facebook_presence,
                            'page_activity': page_activity,
                            'customer_reviews': customer_reviews,
                            'review_count': fb_info.get('review_count', 'unknown'),
                            'local_presence': local_presence,
                            'engagement_indicators': fb_info.get('engagement_indicators', []),
                            'business_information': fb_info.get('business_information', [])
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create Facebook candidate: {e}")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Facebook analysis failed: {e}")
            return []
    
    async def _search_youtube(
        self, 
        target: DiscoveryTarget,
        strategy: SearchStrategy,
        platform_info: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Search YouTube for business channels and content
        
        Args:
            target: Discovery target
            strategy: Search strategy
            platform_info: YouTube-specific information
            
        Returns:
            List of candidates from YouTube analysis
        """
        
        youtube_prompt = f"""
        Analyze YouTube business presence for {target.service_category} providers in {target.country}.
        
        YouTube Content Analysis:
        - Service Category: {target.service_category}
        - Country: {target.country}
        - Language: {target.language}
        
        YouTube discovery patterns:
        1. **Business Channels**: Official company YouTube channels
        2. **Educational Content**: Companies providing industry education
        3. **Case Studies**: Channels showcasing client work and results
        4. **Product Demos**: Companies demonstrating their services
        5. **Thought Leadership**: Industry experts and company executives
        
        Look for providers with:
        - Professional business channels
        - Regular educational content
        - Client testimonials and case studies
        - Product/service demonstrations
        - Industry thought leadership content
        
        Return JSON with YouTube-discovered providers:
        [{{
          "company_name": "business_name",
          "youtube_channel": "channel_name",
          "estimated_website": "website_url",
          "subscriber_count": "estimated_subscribers",
          "content_quality": "professional/amateur",
          "content_types": ["tutorials", "case_studies", "demos", "testimonials"],
          "upload_frequency": "regular/occasional/rare",
          "engagement_level": "high/medium/low",
          "business_focus": ["education", "marketing", "thought_leadership"]
        }}]
        
        Focus on 5-6 channels with strongest business content and relevance.
        """
        
        try:
            response = await self.ask_ai(
                prompt=youtube_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=800
            )
            
            youtube_data = safe_json_parse(response, default=[])
            
            candidates = []
            for yt_info in youtube_data:
                try:
                    # Calculate confidence based on YouTube business indicators
                    content_quality = yt_info.get('content_quality', '').lower()
                    upload_frequency = yt_info.get('upload_frequency', '').lower()
                    engagement_level = yt_info.get('engagement_level', '').lower()
                    content_types = yt_info.get('content_types', [])
                    
                    base_confidence = 0.3  # Lower base for YouTube (content-focused)
                    
                    # Boost confidence for professional content
                    if content_quality == 'professional':
                        base_confidence += 0.2
                    
                    # Boost for regular uploads
                    if upload_frequency == 'regular':
                        base_confidence += 0.15
                    elif upload_frequency == 'occasional':
                        base_confidence += 0.1
                    
                    # Boost for high engagement
                    if engagement_level == 'high':
                        base_confidence += 0.15
                    elif engagement_level == 'medium':
                        base_confidence += 0.1
                    
                    # Boost for business-relevant content types
                    business_content_count = sum(1 for ct in content_types if ct in ['case_studies', 'demos', 'testimonials'])
                    content_boost = min(0.15, business_content_count * 0.05)
                    
                    final_confidence = min(0.7, base_confidence + content_boost)
                    
                    candidate = ProviderCandidate(
                        name=yt_info.get('company_name', 'Unknown'),
                        website=yt_info.get('estimated_website', ''),
                        discovery_method='social_youtube_content_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='content_creator',
                        ai_analysis={
                            'source': 'youtube_analysis',
                            'youtube_channel': yt_info.get('youtube_channel', ''),
                            'subscriber_count': yt_info.get('subscriber_count', 'unknown'),
                            'content_quality': content_quality,
                            'content_types': content_types,
                            'upload_frequency': upload_frequency,
                            'engagement_level': engagement_level,
                            'business_focus': yt_info.get('business_focus', [])
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create YouTube candidate: {e}")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"YouTube analysis failed: {e}")
            return []
    
    async def _search_platform_via_ai(
        self, 
        platform_name: str,
        target: DiscoveryTarget,
        strategy: SearchStrategy
    ) -> List[ProviderCandidate]:
        """
        Use AI knowledge to search unknown or specialized platforms
        
        Args:
            platform_name: Name of the platform
            target: Discovery target
            strategy: Search strategy
            
        Returns:
            List of candidates from AI knowledge of the platform
        """
        
        platform_prompt = f"""
        Analyze {platform_name} for {target.service_category} providers in {target.country}.
        
        Platform Analysis Context:
        - Platform: {platform_name}
        - Service Category: {target.service_category}
        - Country: {target.country}
        - Language: {target.language}
        
        If you have knowledge about {platform_name}, identify providers that would likely have presence there.
        
        Consider:
        - Platform's primary user base and business focus
        - How {target.service_category} providers typically use this platform
        - Regional popularity of {platform_name} in {target.country}
        - Business discovery potential on this platform
        
        Return JSON with platform-specific provider insights:
        [{{
          "company_name": "provider_name",
          "estimated_website": "website_url",
          "platform_presence": "strong/moderate/emerging",
          "presence_reasoning": "why_they_would_be_on_this_platform",
          "discovery_confidence": "high/medium/low"
        }}]
        
        If you don't have specific knowledge about {platform_name}, return an empty array.
        Focus on 3-5 most relevant providers if the platform is known.
        """
        
        try:
            response = await self.ask_ai(
                prompt=platform_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.SIMPLE,
                max_tokens=600
            )
            
            platform_data = safe_json_parse(response, default=[])
            
            candidates = []
            for provider_info in platform_data:
                try:
                    discovery_confidence = provider_info.get('discovery_confidence', '').lower()
                    platform_presence = provider_info.get('platform_presence', '').lower()
                    
                    # Base confidence for AI knowledge discovery
                    base_confidence = 0.3
                    
                    # Adjust based on discovery confidence
                    if discovery_confidence == 'high':
                        base_confidence += 0.2
                    elif discovery_confidence == 'medium':
                        base_confidence += 0.1
                    
                    # Adjust based on platform presence
                    if platform_presence == 'strong':
                        base_confidence += 0.15
                    elif platform_presence == 'moderate':
                        base_confidence += 0.1
                    
                    final_confidence = min(0.65, base_confidence)
                    
                    candidate = ProviderCandidate(
                        name=provider_info.get('company_name', 'Unknown'),
                        website=provider_info.get('estimated_website', ''),
                        discovery_method=f'social_{platform_name}_ai_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='platform_specific',
                        ai_analysis={
                            'source': f'{platform_name}_ai_analysis',
                            'platform_presence': platform_presence,
                            'presence_reasoning': provider_info.get('presence_reasoning', ''),
                            'discovery_confidence': discovery_confidence
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create {platform_name} candidate: {e}")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"{platform_name} AI analysis failed: {e}")
            return []
    
    async def _multi_platform_discovery(
        self, 
        target: DiscoveryTarget,
        social_landscape: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Execute multi-platform discovery strategy
        
        Args:
            target: Discovery target
            social_landscape: Social landscape analysis
            
        Returns:
            List of candidates from multi-platform analysis
        """
        self.logger.info("🌐 Executing multi-platform discovery strategy")
        
        # This method would coordinate discovery across multiple platforms
        # For now, we return an empty list as the individual platform methods handle discovery
        return []
    
    async def _analyze_industry_communities(
        self, 
        target: DiscoveryTarget,
        social_landscape: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Analyze industry-specific communities and forums
        
        Args:
            target: Discovery target
            social_landscape: Social landscape analysis
            
        Returns:
            List of candidates from community analysis
        """
        
        # Determine relevant industry communities
        industry_category = self._categorize_industry(target.service_category)
        relevant_communities = self.industry_communities.get(industry_category, [])
        
        community_prompt = f"""
        Analyze industry communities and forums for {target.service_category} providers in {target.country}.
        
        Community Analysis Context:
        - Service Category: {target.service_category}
        - Industry Category: {industry_category}
        - Country: {target.country}
        - Relevant Communities: {relevant_communities}
        
        Community Discovery Focus:
        1. **Active Contributors**: Companies/individuals actively contributing to discussions
        2. **Solution Providers**: Businesses mentioned as solutions to user problems
        3. **Expert Recognition**: Companies recognized as industry experts
        4. **Recommendation Patterns**: Frequently recommended service providers
        5. **Thought Leadership**: Organizations sharing valuable industry insights
        
        Analyze community presence patterns:
        - Companies with team members actively participating
        - Businesses frequently recommended by community members
        - Organizations hosting community events or discussions
        - Providers mentioned in solution threads
        - Companies sharing valuable resources or insights
        
        Return JSON with community-discovered providers:
        [{{
          "company_name": "provider_name",
          "estimated_website": "website_url",
          "community_presence": ["community_names"],
          "participation_type": "active_contributor/solution_provider/thought_leader",
          "community_reputation": "strong/moderate/emerging",
          "expertise_areas": ["specific_knowledge_areas"],
          "community_endorsements": ["types_of_community_validation"],
          "discovery_reasoning": "why_this_indicates_legitimate_provider"
        }}]
        
        Focus on 4-6 providers with strongest community validation.
        """
        
        try:
            response = await self.ask_ai(
                prompt=community_prompt,
                provider="ollama",
                task_complexity=TaskComplexity.MEDIUM,
                max_tokens=1000
            )
            
            community_data = safe_json_parse(response, default=[])
            
            candidates = []
            for community_info in community_data:
                try:
                    # Calculate confidence based on community validation
                    participation_type = community_info.get('participation_type', '')
                    community_reputation = community_info.get('community_reputation', '').lower()
                    community_endorsements = community_info.get('community_endorsements', [])
                    
                    base_confidence = 0.5  # Medium base for community discovery
                    
                    # Boost confidence based on participation type
                    if 'thought_leader' in participation_type:
                        base_confidence += 0.2
                    elif 'active_contributor' in participation_type:
                        base_confidence += 0.15
                    elif 'solution_provider' in participation_type:
                        base_confidence += 0.1
                    
                    # Boost for strong reputation
                    if community_reputation == 'strong':
                        base_confidence += 0.15
                    elif community_reputation == 'moderate':
                        base_confidence += 0.1
                    
                    # Boost for endorsements
                    endorsement_boost = min(0.15, len(community_endorsements) * 0.05)
                    final_confidence = min(0.8, base_confidence + endorsement_boost)
                    
                    candidate = ProviderCandidate(
                        name=community_info.get('company_name', 'Unknown'),
                        website=community_info.get('estimated_website', ''),
                        discovery_method='social_community_analysis',
                        confidence_score=final_confidence,
                        business_category=target.service_category,
                        market_position='community_recognized',
                        ai_analysis={
                            'source': 'community_analysis',
                            'community_presence': community_info.get('community_presence', []),
                            'participation_type': participation_type,
                            'community_reputation': community_reputation,
                            'expertise_areas': community_info.get('expertise_areas', []),
                            'community_endorsements': community_endorsements,
                            'discovery_reasoning': community_info.get('discovery_reasoning', '')
                        }
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create community candidate: {e}")
            
            self.intelligence_stats['community_mentions'] += len(candidates)
            
            self.logger.info(f"👥 Community analysis: {len(candidates)} candidates found")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Community analysis failed: {e}")
            return []
    
    def _categorize_industry(self, service_category: str) -> str:
        """
        Categorize service category into broader industry groups
        
        Args:
            service_category: Specific service category
            
        Returns:
            Broader industry category
        """
        service_lower = service_category.lower()
        
        if any(tech_term in service_lower for tech_term in ['software', 'it', 'cloud', 'tech', 'digital', 'ai', 'data']):
            return 'technology'
        elif any(biz_term in service_lower for biz_term in ['consulting', 'business', 'management', 'strategy', 'marketing']):
            return 'business'
        elif any(fin_term in service_lower for fin_term in ['finance', 'accounting', 'banking', 'investment', 'insurance']):
            return 'finance'
        elif any(health_term in service_lower for health_term in ['health', 'medical', 'healthcare', 'wellness', 'telemedicine']):
            return 'healthcare'
        elif any(edu_term in service_lower for edu_term in ['education', 'training', 'learning', 'academic', 'teaching']):
            return 'education'
        else:
            return 'business'  # Default fallback
    
    async def _validate_social_candidates(
        self, 
        candidates: List[ProviderCandidate],
        target: DiscoveryTarget,
        social_landscape: Dict[str, Any]
    ) -> List[ProviderCandidate]:
        """
        Validate and enhance social intelligence candidates
        
        Args:
            candidates: List of candidates from social intelligence
            target: Discovery target
            social_landscape: Social landscape analysis
            
        Returns:
            List of validated candidates
        """
        if not candidates:
            return []
        
        self.logger.info(f"🔍 Validating {len(candidates)} social intelligence candidates")
        
        # Remove duplicates by name (case-insensitive)
        unique_candidates = {}
        for candidate in candidates:
            normalized_name = candidate.name.lower().strip()
            if normalized_name not in unique_candidates:
                unique_candidates[normalized_name] = candidate
            else:
                # Keep candidate with higher confidence or more social signals
                existing = unique_candidates[normalized_name]
                if (candidate.confidence_score > existing.confidence_score or 
                    len(candidate.ai_analysis.get('social_signals', [])) > len(existing.ai_analysis.get('social_signals', []))):
                    unique_candidates[normalized_name] = candidate
        
        validated_candidates = list(unique_candidates.values())
        
        # Add social validation context
        for candidate in validated_candidates:
            try:
                # Calculate social validation score
                social_score = self._calculate_social_validation_score(candidate, social_landscape)
                
                # Add social context to AI analysis
                candidate.ai_analysis['social_validation'] = {
                    'validation_score': social_score,
                    'social_signals_count': len([k for k in candidate.ai_analysis.keys() if 'social' in k.lower()]),
                    'platform_diversity': self._count_platform_diversity(candidate),
                    'validation_timestamp': time.time()
                }
                
                # Adjust confidence based on social validation
                validation_adjustment = (social_score - 0.5) * 0.1  # +/- 10% based on social validation
                candidate.confidence_score = max(0.1, min(0.9, candidate.confidence_score + validation_adjustment))
                
            except Exception as e:
                self.logger.warning(f"Failed to add social validation for {candidate.name}: {e}")
        
        # Filter by minimum confidence threshold
        final_candidates = [
            c for c in validated_candidates 
            if c.confidence_score >= target.min_confidence_score
        ]
        
        # Sort by confidence score
        final_candidates.sort(key=lambda c: c.confidence_score, reverse=True)
        
        self.logger.info(f"✅ Social validation complete: {len(final_candidates)} high-confidence candidates")
        return final_candidates
    
    def _calculate_social_validation_score(self, candidate: ProviderCandidate, social_landscape: Dict[str, Any]) -> float:
        """
        Calculate social validation score based on multiple factors
        
        Args:
            candidate: Provider candidate
            social_landscape: Social landscape context
            
        Returns:
            Social validation score (0.0-1.0)
        """
        score = 0.5  # Base score
        
        # Factor 1: Platform presence diversity
        platform_count = self._count_platform_diversity(candidate)
        score += min(0.2, platform_count * 0.05)
        
        # Factor 2: Discovery method credibility
        method = candidate.discovery_method
        if 'linkedin' in method:
            score += 0.15  # LinkedIn is more business-credible
        elif 'reddit' in method or 'community' in method:
            score += 0.1   # Community validation is valuable
        elif 'twitter' in method or 'facebook' in method:
            score += 0.05  # General social media
        
        # Factor 3: AI analysis depth
        ai_analysis_keys = len(candidate.ai_analysis.keys())
        score += min(0.1, ai_analysis_keys * 0.02)
        
        # Factor 4: Market position indicators
        if candidate.market_position in ['professional_network', 'community_recognized']:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _count_platform_diversity(self, candidate: ProviderCandidate) -> int:
        """
        Count how many different platforms/sources contributed to this candidate
        
        Args:
            candidate: Provider candidate
            
        Returns:
            Number of diverse platform sources
        """
        method = candidate.discovery_method.lower()
        platforms = ['linkedin', 'twitter', 'facebook', 'reddit', 'youtube', 'community']
        
        return sum(1 for platform in platforms if platform in method)
    
    async def _generate_default_social_strategy(self, target: DiscoveryTarget) -> List[SearchStrategy]:
        """Generate default social intelligence strategy when none provided"""
        
        default_strategy = SearchStrategy(
            method="social_intelligence",
            priority=6,
            queries=[
                f"{target.service_category} providers {target.country}",
                f"best {target.service_category} {target.country}",
                f"{target.service_category} recommendations {target.country}"
            ],
            platforms=["linkedin", "twitter", "reddit", "facebook"],
            expected_yield="8-15",
            ai_analysis_needed=True,
            follow_up_actions=["social_sentiment_analysis", "community_validation"],
            metadata={"auto_generated": True, "multi_platform": True}
        )
        
        return [default_strategy]
    
    def _get_fallback_social_landscape(self, target: DiscoveryTarget) -> Dict[str, Any]:
        """Get basic social landscape when AI analysis fails"""
        
        return {
            "platform_priorities": [
                {
                    "platform": "linkedin",
                    "business_usage": "high",
                    "discovery_potential": "high",
                    "content_types": ["company_pages", "executive_profiles"],
                    "search_strategies": ["company_search", "people_search"]
                },
                {
                    "platform": "reddit",
                    "business_usage": "medium",
                    "discovery_potential": "medium",
                    "content_types": ["recommendations", "discussions"],
                    "search_strategies": ["subreddit_search", "keyword_search"]
                }
            ],
            "industry_communities": [
                {
                    "community_name": f"{target.service_category} professionals",
                    "platform": "linkedin",
                    "member_count": "unknown",
                    "relevance": "high",
                    "access_level": "public"
                }
            ],
            "recommended_approach": "focus_on_professional_networks"
        }
    
    async def _get_cached_social_data(self, cache_key: str) -> Optional[List[ProviderCandidate]]:
        """Retrieve cached social intelligence data"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(f"social:{cache_key}")
            if cached_data:
                candidates_data = json.loads(cached_data)
                return [ProviderCandidate(**data) for data in candidates_data]
        except Exception as e:
            self.logger.warning(f"Failed to get cached social data: {e}")
        
        return None
    
    async def _cache_social_data(self, cache_key: str, candidates: List[ProviderCandidate]):
        """Cache social intelligence data for reuse"""
        if not self.redis_client:
            return
        
        try:
            candidates_data = [candidate.to_dict() for candidate in candidates]
            await self.redis_client.setex(
                f"social:{cache_key}",
                self.config