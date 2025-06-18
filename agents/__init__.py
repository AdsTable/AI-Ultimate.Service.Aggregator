"""
Multi-Agent AI System for Autonomous Service Discovery and Aggregation

This package contains specialized AI agents for:
- Market discovery and provider identification
- Intelligent content extraction and analysis  
- Market intelligence and competitive analysis

All agents integrate with the existing AIAsyncClient for cost-optimized AI usage.
"""

from .base import BaseAgent, AgentError, AgentMetrics
from .discovery import AdvancedDiscoveryOrchestrator, DiscoveryTarget, ProviderCandidate
from .extraction import IntelligentContentExtractor, ExtractionResult
from .intelligence import MarketIntelligenceAgent, MarketIntelligence

__version__ = "1.0.0"
__all__ = [
    "BaseAgent",
    "AgentError", 
    "AgentMetrics",
    "AdvancedDiscoveryOrchestrator",
    "DiscoveryTarget",
    "ProviderCandidate",
    "IntelligentContentExtractor", 
    "ExtractionResult",
    "MarketIntelligenceAgent",
    "MarketIntelligence"
]