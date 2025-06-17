# services/review_aggregation.py
class AutonomousReviewAggregator:
    """AI-powered review discovery and sentiment analysis across platforms"""
    
    def __init__(self):
        self.review_sources = {
            "trustpilot": TrustpilotScraper(),
            "google_reviews": GoogleReviewsScraper(),
            "facebook": FacebookReviewsScraper(),
            "reddit": RedditScraper(),
            "local_forums": NorwegianForumsScraper(),
            "youtube": YouTubeCommentsScraper()
        }
        
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        
    async def discover_and_aggregate_reviews(self, 
                                           provider_name: str,
                                           country: str = "Norway") -> Dict[str, Any]:
        """
        Comprehensive review discovery and analysis
        """
        
        # AI-generated search variations for the provider
        search_variations = await self._generate_search_variations(provider_name)
        
        aggregated_reviews = {}
        
        for source_name, scraper in self.review_sources.items():
            try:
                reviews = await scraper.extract_reviews(
                    provider_name=provider_name,
                    search_variations=search_variations,
                    country=country
                )
                
                # AI-powered sentiment and authenticity analysis
                processed_reviews = await self._process_reviews(reviews, source_name)
                
                aggregated_reviews[source_name] = processed_reviews
                
            except Exception as e:
                logger.warning(f"Review extraction failed for {source_name}: {e}")
                continue
        
        # Generate comprehensive insights
        return await self._generate_review_insights(aggregated_reviews)
    
    async def _process_reviews(self, reviews: List[Dict], source: str) -> Dict[str, Any]:
        """Advanced review processing with AI analysis"""
        
        processed = {
            "total_reviews": len(reviews),
            "average_rating": 0,
            "sentiment_distribution": {},
            "key_themes": [],
            "authenticity_score": 0,
            "temporal_trends": {},
            "reviews": []
        }
        
        if not reviews:
            return processed
        
        # Sentiment analysis
        sentiments = []
        for review in reviews:
            sentiment = self.sentiment_analyzer(review.get("text", ""))
            sentiments.append({
                **review,
                "ai_sentiment": sentiment[0]["label"],
                "confidence": sentiment[0]["score"]
            })
        
        # Authenticity scoring using AI
        authenticity_scores = await self._score_review_authenticity(sentiments)
        
        # Theme extraction
        themes = await self._extract_key_themes(sentiments)
        
        processed.update({
            "reviews": sentiments,
            "authenticity_score": np.mean(authenticity_scores),
            "key_themes": themes,
            "average_rating": np.mean([r.get("rating", 0) for r in reviews])
        })
        
        return processed