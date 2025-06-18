"""
Discovery agents for autonomous market research and provider identification

This module implements a comprehensive multi-agent discovery system that can
autonomously find service providers across different markets using various
discovery methods and AI-powered validation.
"""

from .orchestrator import AdvancedDiscoveryOrchestrator
from .models import DiscoveryTarget, ProviderCandidate, SearchStrategy
from .search_engine_agent import SearchEngineAgent
from .regulatory_scanner import RegulatoryScanner
from .competitor_analyzer import CompetitorAnalyzer
from .social_intelligence_agent import SocialIntelligenceAgent

__all__ = [
    'AdvancedDiscoveryOrchestrator',
    'DiscoveryTarget',
    'ProviderCandidate',
    'SearchStrategy',
    'SearchEngineAgent',
    'RegulatoryScanner',
    'CompetitorAnalyzer',
    'SocialIntelligenceAgent'
]

__version__ = "1.0.0"