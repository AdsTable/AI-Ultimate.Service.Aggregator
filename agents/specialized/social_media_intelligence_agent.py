# agents/specialized/social_media_intelligence_agent.py
"""
Advanced Social Media Intelligence Agent for provider discovery
Analyzes social media platforms, forums, and community discussions
"""
import asyncio
import re
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
from urllib.parse import urljoin, quote

from .base_discovery_agent import BaseDiscoveryAgent, DiscoveryResult

class SocialMediaIntelligenceAgent(BaseDiscoveryAgent):
    """
    Discovers service providers through social media analysis and community intelligence
    Focuses on customer discussions, company presence, and industry conversations
    """
    
    def __init__(self, ai_client, config: Dict[str, Any]):
        super().__init__(ai_client, config)
        
        # Platform configurations with rate limits and search endpoints
        self.platforms = {
            'reddit': {
                'base_url': 'https://www.reddit.com',
                'search_endpoint': '/search.json',
                'rate_limit': 2.0,  # seconds between requests
                'user_agent_required': True
            },
            'hackernews': {
                'base_url': 'https://hn.algolia.com/api/v1',
                'search_endpoint': '/search',
                'rate_limit': 1.0
            },
            'twitter_alternative': {
                # Using web scraping approach instead of API
                'base_url': 'https://nitter.net',
                'search_endpoint': '/search',
                'rate_limit': 3.0
            }
        }
        
        # Search query templates for different service categories
        self.query_templates = {
            'electricity': [
                'electricity provider {country}',
                'power company {country}',
                'energy supplier {country}',
                'strøm leverandør {country}',  # Norwegian
                'best electricity deal {country}'
            ],
            'mobile': [
                'mobile provider {country}',
                'cell phone carrier {country}', 
                'mobilabonnement {country}',  # Norwegian
                'mobil operatør {country}',
                'best mobile plan {country}'
            ],
            'internet': [
                'internet provider {country}',
                'ISP {country}',
                'broadband {country}',
                'fiber internet {country}',
                'internettleverandør {country}'  # Norwegian
            ]
        }
        
    async def _initialize_agent_specific(self):
        """Initialize social media specific resources"""
        logger.info("🔍 Initializing Social Media Intelligence Agent")
        
        # Test connectivity to platforms
        connectivity_results = await self._test_platform_connectivity()
        self.active_platforms = [
            platform for platform, status in connectivity_results.items() 
            if status
        ]
        
        logger.info(f"✅ Active platforms: {', '.join(self.active_platforms)}")
    
    async def _cleanup_agent_specific(self):
        """Cleanup social media specific resources"""
        logger.info("🔄 Cleaning up Social Media Intelligence Agent")
    
    async def discover_providers(self, target_country: str, service_category: str) -> List[DiscoveryResult]:
        """
        Main discovery method for social media intelligence
        """
        logger.info(f"🔍 Starting social media discovery for {service_category} in {target_country}")
        
        # Check cache first
        cache_key = f"social_{target_country}_{service_category}"
        cached_results = await self._get_cached_results(cache_key)
        if cached_results:
            return cached_results
        
        # Generate search queries
        search_queries = self._generate_search_queries(target_country, service_category)
        
        # Discover from all active platforms
        all_results = []
        
        for platform in self.active_platforms:
            try:
                platform_results = await self._discover_from_platform(
                    platform, search_queries, target_country, service_category
                )
                all_results.extend(platform_results)
                logger.info(f"📱 {platform}: Found {len(platform_results)} potential providers")
                
            except Exception as e:
                logger.error(f"❌ Discovery failed for {platform}: {e}")
        
        # AI-powered result validation and enhancement
        validated_results = await self._validate_and_enhance_results(all_results, service_category)
        
        # Cache results
        await self._cache_results(cache_key, validated_results)
        
        logger.info(f"✅ Social media discovery complete: {len(validated_results)} validated providers")
        return validated_results
    
    def _generate_search_queries(self, country: str, service_category: str) -> List[str]:
        """Generate intelligent search queries for social media platforms"""
        
        templates = self.query_templates.get(service_category, [])
        queries = []
        
        for template in templates:
            # Standard country name
            queries.append(template.format(country=country))
            
            # Country variations
            country_variations = {
                'Norway': ['Norge', 'Norwegian', 'NO'],
                'Sweden': ['Sverige', 'Swedish', 'SE'], 
                'Denmark': ['Danmark', 'Danish', 'DK']
            }
            
            for variation in country_variations.get(country, []):
                queries.append(template.format(country=variation))
        
        # Add time-based queries for recent discussions
        time_queries = [
            f'best {service_category} provider {country} 2024',
            f'switching {service_category} {country}',
            f'{service_category} recommendations {country}'
        ]
        queries.extend(time_queries)
        
        return queries
    
    async def _discover_from_platform(self, 
                                    platform: str, 
                                    queries: List[str], 
                                    country: str, 
                                    service_category: str) -> List[DiscoveryResult]:
        """Discover providers from a specific social media platform"""
        
        platform_config = self.platforms[platform]
        results = []
        
        for query in queries[:5]:  # Limit queries per platform
            try:
                await self._apply_rate_limiting(platform, platform_config['rate_limit'])
                
                # Execute platform-specific search
                search_results = await self._execute_platform_search(platform, query)
                
                # Extract provider mentions from results
                provider_mentions = await self._extract_provider_mentions(
                    search_results, query, platform, service_category
                )
                
                results.extend(provider_mentions)
                
            except Exception as e:
                logger.error(f"Search failed on {platform} for query '{query}': {e}")
        
        return results
    
    async def _execute_platform_search(self, platform: str, query: str) -> List[Dict[str, Any]]:
        """Execute search on specific platform"""
        
        platform_config = self.platforms[platform]
        
        if platform == 'reddit':
            return await self._search_reddit(query)
        elif platform == 'hackernews':
            return await self._search_hackernews(query)
        elif platform == 'twitter_alternative':
            return await self._search_twitter_alternative(query)
        else:
            logger.warning(f"Unknown platform: {platform}")
            return []
    
    async def _search_reddit(self, query: str) -> List[Dict[str, Any]]:
        """Search Reddit for provider discussions"""
        
        search_url = f"https://www.reddit.com/search.json?q={quote(query)}&sort=relevance&limit=50"
        
        try:
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    posts = data.get('data', {}).get('children', [])
                    
                    results = []
                    for post in posts:
                        post_data = post.get('data', {})
                        results.append({
                            'title': post_data.get('title', ''),
                            'content': post_data.get('selftext', ''),
                            'url': f"https://reddit.com{post_data.get('permalink', '')}",
                            'subreddit': post_data.get('subreddit', ''),
                            'score': post_data.get('score', 0),
                            'comments': post_data.get('num_comments', 0),
                            'platform': 'reddit'
                        })
                    
                    return results
                else:
                    logger.warning(f"Reddit search failed with status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Reddit search error: {e}")
            return []
    
    async def _search_hackernews(self, query: str) -> List[Dict[str, Any]]:
        """Search Hacker News for provider discussions"""
        
        search_url = f"https://hn.algolia.com/api/v1/search?query={quote(query)}&tags=story&hitsPerPage=30"
        
        try:
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    hits = data.get('hits', [])
                    
                    results = []
                    for hit in hits:
                        results.append({
                            'title': hit.get('title', ''),
                            'content': hit.get('story_text', ''),
                            'url': hit.get('url', ''),
                            'comments': hit.get('num_comments', 0),
                            'points': hit.get('points', 0),
                            'platform': 'hackernews'
                        })
                    
                    return results
                else:
                    logger.warning(f"HackerNews search failed with status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"HackerNews search error: {e}")
            return []
    
    async def _search_twitter_alternative(self, query: str) -> List[Dict[str, Any]]:
        """Search Twitter alternative (Nitter) for provider mentions"""
        
        search_url = f"https://nitter.net/search?f=tweets&q={quote(query)}"
        
        try:
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    
                    # Parse HTML for tweet content (simplified extraction)
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    tweets = soup.find_all('div', class_='tweet-content')
                    results = []
                    
                    for tweet in tweets[:20]:  # Limit results
                        tweet_text = tweet.get_text(strip=True)
                        if len(tweet_text) > 20:  # Filter out too short tweets
                            results.append({
                                'title': tweet_text[:100] + '...' if len(tweet_text) > 100 else tweet_text,
                                'content': tweet_text,
                                'url': '',  # Would need more parsing for actual URLs
                                'platform': 'twitter_alternative'
                            })
                    
                    return results
                else:
                    logger.warning(f"Twitter alternative search failed with status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Twitter alternative search error: {e}")
            return []
    
    async def _extract_provider_mentions(self, 
                                       search_results: List[Dict[str, Any]], 
                                       original_query: str,
                                       platform: str,
                                       service_category: str) -> List[DiscoveryResult]:
        """Extract provider mentions from search results using AI"""
        
        if not search_results:
            return []
        
        # Combine all content for AI analysis
        combined_content = []
        for result in search_results[:10]:  # Limit for AI processing
            content = f"Title: {result.get('title', '')}\nContent: {result.get('content', '')}"
            combined_content.append(content)
        
        full_content = '\n\n---\n\n'.join(combined_content)
        
        # AI analysis prompt
        analysis_prompt = f"""
        Analyze these social media discussions about {service_category} providers to extract company names and websites.
        
        Original Query: {original_query}
        Platform: {platform}
        Content: {full_content[:8000]}  # Limit to avoid token overflow
        
        Extract mentioned service providers and return JSON array:
        [
            {{
                "provider_name": "Company Name",
                "website_hint": "extracted_url_or_null",
                "confidence_score": 0.0-1.0,
                "mention_context": "positive/negative/neutral",
                "evidence": "brief quote supporting identification",
                "business_indicators": ["indicator1", "indicator2"]
            }}
        ]
        
        Focus on legitimate business entities, avoid personal recommendations unless they clearly reference companies.
        """
        
        try:
            response = await self.ai_client.ask(
                prompt=analysis_prompt,
                provider="ollama",
                task_complexity="complex"
            )
            
            ai_extractions = json.loads(response)
            
            # Convert AI extractions to DiscoveryResult objects
            discovery_results = []
            
            for extraction in ai_extractions:
                if extraction.get('confidence_score', 0) > 0.6:
                    provider_name = extraction.get('provider_name', '')
                    website_hint = extraction.get('website_hint', '')
                    
                    # Try to find/construct website URL
                    website_url = await self._resolve_provider_website(provider_name, website_hint)
                    
                    if website_url:
                        result = self._create_discovery_result(
                            provider_name=provider_name,
                            website_url=website_url,
                            confidence_score=extraction.get('confidence_score', 0),
                            discovery_method='social_media_intelligence',
                            source_platform=platform,
                            additional_data={
                                'business_details': {
                                    'mention_context': extraction.get('mention_context'),
                                    'evidence': extraction.get('evidence'),
                                    'business_indicators': extraction.get('business_indicators', [])
                                },
                                'raw_data': {
                                    'original_query': original_query,
                                    'search_results_count': len(search_results),
                                    'ai_extraction': extraction
                                }
                            }
                        )
                        discovery_results.append(result)
            
            return discovery_results
            
        except Exception as e:
            logger.error(f"AI extraction failed: {e}")
            return []
    
    async def _resolve_provider_website(self, provider_name: str, website_hint: str) -> Optional[str]:
        """Attempt to resolve provider website URL"""
        
        # If we have a direct URL hint, validate it
        if website_hint and website_hint.startswith('http'):
            if await self._validate_url(website_hint):
                return website_hint
        
        # Try to construct likely website URLs
        name_cleaned = re.sub(r'[^a-zA-Z0-9]', '', provider_name.lower())
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
    
    async def _validate_url(self, url: str) -> bool:
        """Validate if URL is accessible and contains business content"""
        try:
            async with self.session.head(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status == 200
        except:
            return False
    
    async def _validate_and_enhance_results(self, 
                                          results: List[DiscoveryResult], 
                                          service_category: str) -> List[DiscoveryResult]:
        """Validate and enhance discovery results using AI"""
        
        if not results:
            return []
        
        # Group by provider name to remove duplicates
        unique_providers = {}
        for result in results:
            name_key = result.provider_name.lower().strip()
            if name_key not in unique_providers or result.confidence_score > unique_providers[name_key].confidence_score:
                unique_providers[name_key] = result
        
        validated_results = []
        
        for result in unique_providers.values():
            # Additional validation through website analysis
            if result.website_url:
                website_validation = await self._validate_provider_website(result.website_url, service_category)
                
                # Update confidence based on website validation
                if website_validation.get('is_legitimate_provider', False):
                    result.confidence_score = min(1.0, result.confidence_score + 0.2)
                    result.business_details.update(website_validation.get('business_info', {}))
                    validated_results.append(result)
        
        # Sort by confidence score
        validated_results.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return validated_results[:20]  # Return top 20 results
    
    async def _validate_provider_website(self, website_url: str, service_category: str) -> Dict[str, Any]:
        """Validate provider website using AI analysis"""
        
        try:
            async with self.session.get(website_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    html_content = await response.text()
                    
                    # Extract text content for analysis
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    text_content = soup.get_text(separator=' ', strip=True)[:5000]
                    
                    # AI validation
                    validation = await self._ai_analyze_text(
                        text_content, 
                        'website',
                        {'url': website_url, 'service_category': service_category}
                    )
                    
                    return validation
                else:
                    return {'is_legitimate_provider': False, 'reasoning': f'HTTP {response.status}'}
                    
        except Exception as e:
            return {'is_legitimate_provider': False, 'reasoning': f'Access failed: {str(e)}'}
    
    async def _test_platform_connectivity(self) -> Dict[str, bool]:
        """Test connectivity to all platforms"""
        
        connectivity_results = {}
        
        for platform_name, config in self.platforms.items():
            try:
                test_url = config['base_url']
                async with self.session.get(test_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    connectivity_results[platform_name] = response.status == 200
            except:
                connectivity_results[platform_name] = False
        
        return connectivity_results