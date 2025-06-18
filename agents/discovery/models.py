# agents/discovery/models.py
"""
Data models for discovery agents

This module defines the core data structures used by all discovery agents
for representing discovery targets, provider candidates, and search strategies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime


class TaskComplexity(Enum):
    """Task complexity levels for intelligent AI provider routing"""
    SIMPLE = "simple"        # Basic Q&A, simple analysis
    MEDIUM = "medium"        # Analysis, summarization, content extraction
    COMPLEX = "complex"      # Strategy generation, complex reasoning


@dataclass
class DiscoveryTarget:
    """
    Target specification for autonomous market discovery
    
    This class defines the parameters for a market discovery operation,
    including geographic, linguistic, and regulatory context.
    """
    country: str                              # Target country (e.g., "USA", "Germany")
    service_category: str                     # Service category to discover (e.g., "cloud hosting")
    language: str                            # Primary language for the market (e.g., "English")
    currency: str                            # Local currency (e.g., "USD", "EUR")
    regulatory_bodies: List[str]             # Relevant regulatory organizations
    market_size_estimate: str               # Estimated market size (e.g., "large", "medium")
    discovery_depth: str = "comprehensive"  # Depth of discovery ("basic", "standard", "comprehensive")
    max_providers: int = 50                  # Maximum number of providers to discover
    min_confidence_score: float = 0.7       # Minimum confidence threshold for candidates
    
    def __post_init__(self):
        """Validate target parameters after initialization"""
        if not self.country or not self.service_category:
            raise ValueError("Country and service_category are required fields")
        
        if not isinstance(self.regulatory_bodies, list):
            raise ValueError("regulatory_bodies must be a list")
        
        if self.min_confidence_score < 0.0 or self.min_confidence_score > 1.0:
            raise ValueError("min_confidence_score must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'country': self.country,
            'service_category': self.service_category,
            'language': self.language,
            'currency': self.currency,
            'regulatory_bodies': self.regulatory_bodies,
            'market_size_estimate': self.market_size_estimate,
            'discovery_depth': self.discovery_depth,
            'max_providers': self.max_providers,
            'min_confidence_score': self.min_confidence_score
        }


@dataclass
class ProviderCandidate:
    """
    Discovered provider candidate with AI confidence scoring
    
    This class represents a potential service provider discovered through
    various discovery methods, including AI-generated confidence scores
    and analysis metadata.
    """
    name: str                                    # Company/provider name
    website: str                                # Primary website URL
    discovery_method: str                       # Method used to discover this provider
    confidence_score: float                     # AI confidence score (0.0-1.0)
    business_category: str                      # Business category/industry
    market_position: str                        # Market position (leader/regional/niche/etc.)
    contact_info: Dict[str, Any] = field(default_factory=dict)      # Contact information
    services_preview: List[str] = field(default_factory=list)       # Preview of services offered
    ai_analysis: Dict[str, Any] = field(default_factory=dict)       # AI analysis metadata
    discovered_at: datetime = field(default_factory=datetime.now)   # Discovery timestamp
    
    def __post_init__(self):
        """Validate candidate data after initialization"""
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        
        if not self.name or not self.discovery_method:
            raise ValueError("Name and discovery_method are required fields")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'website': self.website,
            'discovery_method': self.discovery_method,
            'confidence_score': self.confidence_score,
            'business_category': self.business_category,
            'market_position': self.market_position,
            'contact_info': self.contact_info,
            'services_preview': self.services_preview,
            'ai_analysis': self.ai_analysis,
            'discovered_at': self.discovered_at.isoformat()
        }
    
    def update_confidence_score(self, new_score: float, reasoning: str = ""):
        """
        Update confidence score with reasoning
        
        Args:
            new_score: New confidence score (0.0-1.0)
            reasoning: Explanation for the score change
        """
        if not (0.0 <= new_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        
        old_score = self.confidence_score
        self.confidence_score = new_score
        
        # Track confidence score changes
        if 'confidence_history' not in self.ai_analysis:
            self.ai_analysis['confidence_history'] = []
        
        self.ai_analysis['confidence_history'].append({
            'timestamp': datetime.now().isoformat(),
            'old_score': old_score,
            'new_score': new_score,
            'reasoning': reasoning
        })


@dataclass
class SearchStrategy:
    """
    Search strategy configuration for discovery methods
    
    This class defines how a specific discovery method should execute
    its search, including queries, platforms, and execution parameters.
    """
    method: str                                         # Discovery method name
    priority: int                                      # Execution priority (1-10, higher = more important)
    queries: List[str]                                 # Search queries to execute
    platforms: List[str]                              # Platforms/sources to search
    expected_yield: str                               # Expected number of results
    ai_analysis_needed: bool = True                   # Whether AI analysis is required
    follow_up_actions: List[str] = field(default_factory=list)  # Additional actions to take
    metadata: Dict[str, Any] = field(default_factory=dict)      # Additional strategy metadata
    
    def __post_init__(self):
        """Validate strategy data after initialization"""
        if not self.method or not self.queries:
            raise ValueError("Method and queries are required fields")
        
        if not (1 <= self.priority <= 10):
            raise ValueError("Priority must be between 1 and 10")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'method': self.method,
            'priority': self.priority,
            'queries': self.queries,
            'platforms': self.platforms,
            'expected_yield': self.expected_yield,
            'ai_analysis_needed': self.ai_analysis_needed,
            'follow_up_actions': self.follow_up_actions,
            'metadata': self.metadata
        }
    
    def add_query(self, query: str, reason: str = ""):
        """
        Add a new query to the strategy
        
        Args:
            query: Search query to add
            reason: Reason for adding this query
        """
        if query not in self.queries:
            self.queries.append(query)
            
            # Track query additions
            if 'query_history' not in self.metadata:
                self.metadata['query_history'] = []
            
            self.metadata['query_history'].append({
                'timestamp': datetime.now().isoformat(),
                'action': 'added',
                'query': query,
                'reason': reason
            })
    
    def remove_query(self, query: str, reason: str = ""):
        """
        Remove a query from the strategy
        
        Args:
            query: Search query to remove
            reason: Reason for removing this query
        """
        if query in self.queries:
            self.queries.remove(query)
            
            # Track query removals
            if 'query_history' not in self.metadata:
                self.metadata['query_history'] = []
            
            self.metadata['query_history'].append({
                'timestamp': datetime.now().isoformat(),
                'action': 'removed',
                'query': query,
                'reason': reason
            })


@dataclass
class DiscoverySession:
    """
    Represents a complete discovery session with results and metadata
    
    This class encapsulates an entire discovery operation, including
    the target, strategies used, results obtained, and performance metrics.
    """
    session_id: str                                    # Unique session identifier
    target: DiscoveryTarget                           # Discovery target
    strategies_used: List[SearchStrategy]             # Strategies executed
    candidates_found: List[ProviderCandidate]        # Discovered candidates
    start_time: datetime                              # Session start time
    end_time: Optional[datetime] = None               # Session end time
    success: bool = False                             # Whether session was successful
    error_message: Optional[str] = None               # Error message if failed
    performance_metrics: Dict[str, Any] = field(default_factory=dict)  # Performance data
    
    def __post_init__(self):
        """Validate session data after initialization"""
        if not self.session_id or not self.target:
            raise ValueError("session_id and target are required fields")
    
    def complete_session(self, success: bool = True, error_message: Optional[str] = None):
        """
        Mark the discovery session as complete
        
        Args:
            success: Whether the session completed successfully
            error_message: Error message if the session failed
        """
        self.end_time = datetime.now()
        self.success = success
        self.error_message = error_message
        
        # Calculate session duration
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.performance_metrics['duration_seconds'] = duration
        
        # Calculate basic performance metrics
        self.performance_metrics.update({
            'total_candidates': len(self.candidates_found),
            'strategies_executed': len(self.strategies_used),
            'avg_confidence_score': (
                sum(c.confidence_score for c in self.candidates_found) / len(self.candidates_found)
                if self.candidates_found else 0.0
            ),
            'high_confidence_candidates': len([
                c for c in self.candidates_found if c.confidence_score >= 0.8
            ]),
            'discovery_methods_used': list(set(c.discovery_method for c in self.candidates_found))
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'target': self.target.to_dict(),
            'strategies_used': [s.to_dict() for s in self.strategies_used],
            'candidates_found': [c.to_dict() for c in self.candidates_found],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'success': self.success,
            'error_message': self.error_message,
            'performance_metrics': self.performance_metrics
        }
    
    @property
    def duration(self) -> Optional[float]:
        """Get session duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def candidates_by_method(self) -> Dict[str, List[ProviderCandidate]]:
        """Group candidates by discovery method"""
        method_groups = {}
        for candidate in self.candidates_found:
            method = candidate.discovery_method
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(candidate)
        return method_groups
    
    @property
    def success_rate_by_method(self) -> Dict[str, float]:
        """Calculate success rate for each discovery method"""
        method_groups = self.candidates_by_method
        success_rates = {}
        
        for method, candidates in method_groups.items():
            if candidates:
                high_confidence = len([c for c in candidates if c.confidence_score >= 0.7])
                success_rates[method] = high_confidence / len(candidates)
            else:
                success_rates[method] = 0.0
        
        return success_rates