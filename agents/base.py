# agents/base.py
"""
Base Agent Classes and Utilities

Provides foundational classes and utilities for all discovery and extraction agents
in the AI-Ultimate Service Aggregator system.
"""

import asyncio
import logging
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable
from functools import wraps
import aiohttp
from enum import Enum

from services.ai_async_client import AIAsyncClient, TaskComplexity

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Custom exception for agent-related errors"""
    
    def __init__(self, agent_name: str, message: str, original_error: Optional[Exception] = None):
        self.agent_name = agent_name
        self.original_error = original_error
        super().__init__(f"[{agent_name}] {message}")


class OperationStatus(Enum):
    """Status codes for agent operations"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    AI_ERROR = "ai_error"


@dataclass
class AgentConfig:
    """Configuration for agent behavior and performance"""
    name: str
    max_retries: int = 3
    rate_limit: float = 10.0  # requests per second
    preferred_ai_provider: str = "ollama"
    task_complexity: TaskComplexity = TaskComplexity.MEDIUM
    cache_ttl: int = 3600  # cache time-to-live in seconds
    debug: bool = False
    timeout: float = 30.0
    min_confidence_score: float = 0.5
    
    # Advanced configuration
    fallback_providers: List[str] = field(default_factory=lambda: ["huggingface", "groq", "openai"])
    max_concurrent_operations: int = 5
    health_check_interval: int = 300  # 5 minutes
    performance_threshold: float = 0.8  # 80% success rate threshold
    
    # Cost optimization settings
    cost_optimization_enabled: bool = True
    max_cost_per_operation: float = 0.01  # $0.01 max per operation
    prefer_free_providers: bool = True


@dataclass
class OperationMetrics:
    """Metrics for a single operation"""
    operation_id: str
    agent_name: str
    operation_type: str
    status: OperationStatus
    start_time: float
    end_time: float
    execution_time: float
    ai_provider_used: str
    ai_calls_made: int
    tokens_used: int
    estimated_cost: float
    confidence_score: float
    error_message: Optional[str] = None


def safe_json_parse(json_string: str, default: Any = None) -> Any:
    """Safely parse JSON string with fallback"""
    try:
        if isinstance(json_string, str):
            # Handle potential AI response formatting issues
            json_string = json_string.strip()
            if json_string.startswith('```json'):
                json_string = json_string[7:]
            if json_string.endswith('```'):
                json_string = json_string[:-3]
            json_string = json_string.strip()
            
            return json.loads(json_string)
        return json_string
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON parse failed: {e}")
        if default is not None:
            return default
        # Try to extract JSON from partial response
        try:
            # Look for JSON-like patterns
            import re
            json_match = re.search(r'\{.*\}', json_string, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {}


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, exponential: bool = True):
    """Decorator for retrying operations with exponential backoff"""
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        if exponential:
                            delay = base_delay * (2 ** attempt)
                        else:
                            delay = base_delay
                        
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed: {e}")
                        break
            
            raise last_exception
        
        return wrapper
    return decorator


def rate_limiter(max_calls: int, time_window: int):
    """Decorator for rate limiting function calls"""
    
    call_times = []
    lock = asyncio.Lock()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with lock:
                current_time = time.time()
                
                # Remove old calls outside the time window
                call_times[:] = [t for t in call_times if current_time - t < time_window]
                
                # Check if we're at the rate limit
                if len(call_times) >= max_calls:
                    sleep_time = time_window - (current_time - call_times[0]) + 0.1
                    if sleep_time > 0:
                        logger.debug(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                        await asyncio.sleep(sleep_time)
                        call_times.clear()
                
                # Record this call
                call_times.append(current_time)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class PerformanceMonitor:
    """Monitor and track agent performance metrics"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.metrics_history: List[OperationMetrics] = []
        self.start_time = time.time()
        
    def record_operation(self, metrics: OperationMetrics):
        """Record operation metrics"""
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 operations to prevent memory issues
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_performance_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get performance summary for specified time window"""
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics_history 
            if current_time - m.start_time <= time_window
        ]
        
        if not recent_metrics:
            return {
                'total_operations': 0,
                'success_rate': 0.0,
                'average_execution_time': 0.0,
                'total_cost': 0.0,
                'ai_calls_total': 0,
                'average_confidence': 0.0
            }
        
        successful_ops = [m for m in recent_metrics if m.status == OperationStatus.SUCCESS]
        
        return {
            'total_operations': len(recent_metrics),
            'successful_operations': len(successful_ops),
            'success_rate': len(successful_ops) / len(recent_metrics) * 100,
            'average_execution_time': sum(m.execution_time for m in recent_metrics) / len(recent_metrics),
            'total_cost': sum(m.estimated_cost for m in recent_metrics),
            'ai_calls_total': sum(m.ai_calls_made for m in recent_metrics),
            'average_confidence': sum(m.confidence_score for m in successful_ops) / max(1, len(successful_ops)),
            'error_distribution': self._get_error_distribution(recent_metrics),
            'provider_usage': self._get_provider_usage(recent_metrics)
        }
    
    def _get_error_distribution(self, metrics: List[OperationMetrics]) -> Dict[str, int]:
        """Get distribution of error types"""
        error_counts = {}
        for metric in metrics:
            if metric.status != OperationStatus.SUCCESS:
                error_type = metric.error_message.split(':')[0] if metric.error_message else 'Unknown'
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts
    
    def _get_provider_usage(self, metrics: List[OperationMetrics]) -> Dict[str, int]:
        """Get AI provider usage statistics"""
        provider_counts = {}
        for metric in metrics:
            provider = metric.ai_provider_used
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        return provider_counts


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system
    
    Provides common functionality for:
    - AI client integration with fallback providers
    - Error handling and retries
    - Performance tracking and monitoring
    - Configuration management
    - Cost optimization
    - Health monitoring
    """
    
    def __init__(self, config: AgentConfig, ai_client: AIAsyncClient):
        self.config = config
        self.ai_client = ai_client
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(config.name)
        
        # Rate limiting state
        self._last_call_time = 0.0
        self._call_count = 0
        self._rate_limit_lock = asyncio.Lock()
        
        # Health monitoring
        self._health_status = "healthy"
        self._last_health_check = time.time()
        
        # Concurrent operation limiting
        self._operation_semaphore = asyncio.Semaphore(config.max_concurrent_operations)
        
        # Cost tracking
        self._operation_costs = []
        
        # Provider performance tracking
        self._provider_performance = {}
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._setup_agent()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._cleanup_agent()
    
    @abstractmethod
    async def _setup_agent(self) -> None:
        """Initialize agent-specific resources"""
        pass
    
    @abstractmethod
    async def _cleanup_agent(self) -> None:
        """Cleanup agent-specific resources"""
        pass
    
    async def ask_ai(
        self, 
        prompt: str, 
        provider: Optional[str] = None,
        task_complexity: Optional[TaskComplexity] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Ask AI with agent-specific configuration and error handling
        
        Args:
            prompt: The prompt to send to AI
            provider: AI provider override
            task_complexity: Task complexity override
            max_tokens: Maximum tokens override
            temperature: Temperature override
            
        Returns:
            AI response content
        """
        
        operation_id = f"{self.config.name}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Apply rate limiting
        await self._apply_rate_limiting()
        
        # Use agent defaults or overrides
        actual_provider = provider or self.config.preferred_ai_provider
        actual_complexity = task_complexity or self.config.task_complexity
        
        # Apply concurrent operation limiting
        async with self._operation_semaphore:
            try:
                # Health check before operation
                await self._check_health()
                
                # Cost check before operation
                if self.config.cost_optimization_enabled:
                    estimated_cost = await self._estimate_operation_cost(prompt, actual_provider)
                    if estimated_cost > self.config.max_cost_per_operation:
                        # Try to find cheaper provider
                        actual_provider = await self._find_cost_effective_provider(prompt, actual_complexity)
                
                response = await self.ai_client.ask(
                    prompt=prompt,
                    provider=actual_provider,
                    task_complexity=actual_complexity,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Record successful operation
                end_time = time.time()
                metrics = OperationMetrics(
                    operation_id=operation_id,
                    agent_name=self.config.name,
                    operation_type="ai_query",
                    status=OperationStatus.SUCCESS,
                    start_time=start_time,
                    end_time=end_time,
                    execution_time=end_time - start_time,
                    ai_provider_used=actual_provider,
                    ai_calls_made=1,
                    tokens_used=len(prompt.split()) + len(response.split()),  # Rough estimate
                    estimated_cost=await self._estimate_operation_cost(prompt, actual_provider),
                    confidence_score=1.0  # Default high confidence for successful calls
                )
                
                self.performance_monitor.record_operation(metrics)
                self._update_provider_performance(actual_provider, True, end_time - start_time)
                
                return response
                
            except Exception as e:
                # Record failed operation
                end_time = time.time()
                error_message = str(e)
                
                # Determine operation status based on error type
                if "timeout" in error_message.lower():
                    status = OperationStatus.TIMEOUT
                elif "rate" in error_message.lower():
                    status = OperationStatus.RATE_LIMITED
                elif "ai" in error_message.lower() or "provider" in error_message.lower():
                    status = OperationStatus.AI_ERROR
                else:
                    status = OperationStatus.FAILURE
                
                metrics = OperationMetrics(
                    operation_id=operation_id,
                    agent_name=self.config.name,
                    operation_type="ai_query",
                    status=status,
                    start_time=start_time,
                    end_time=end_time,
                    execution_time=end_time - start_time,
                    ai_provider_used=actual_provider,
                    ai_calls_made=1,
                    tokens_used=len(prompt.split()),
                    estimated_cost=0.0,
                    confidence_score=0.0,
                    error_message=error_message
                )
                
                self.performance_monitor.record_operation(metrics)
                self._update_provider_performance(actual_provider, False, end_time - start_time)
                
                # Try fallback providers if configured
                if self.config.fallback_providers and actual_provider in self.config.fallback_providers:
                    return await self._try_fallback_providers(prompt, actual_provider, task_complexity, max_tokens, temperature)
                
                raise AgentError(self.config.name, f"AI operation failed: {error_message}", e)
    
    async def _apply_rate_limiting(self):
        """Apply rate limiting based on agent configuration"""
        async with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_call_time
            min_interval = 1.0 / self.config.rate_limit
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
            
            self._last_call_time = time.time()
    
    async def _check_health(self):
        """Check agent health and update status"""
        current_time = time.time()
        if current_time - self._last_health_check < self.config.health_check_interval:
            return
        
        try:
            # Get recent performance metrics
            performance = self.performance_monitor.get_performance_summary(time_window=3600)
            
            # Determine health based on success rate
            if performance['success_rate'] < self.config.performance_threshold * 100:
                self._health_status = "degraded"
                self.logger.warning(f"Agent health degraded: {performance['success_rate']:.1f}% success rate")
            else:
                self._health_status = "healthy"
            
            self._last_health_check = current_time
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = "unknown"
    
    async def _estimate_operation_cost(self, prompt: str, provider: str) -> float:
        """Estimate cost for an AI operation"""
        # This is a simplified cost estimation
        # In production, you'd integrate with actual provider pricing APIs
        
        token_count = len(prompt.split()) * 1.3  # Rough estimation including response
        
        cost_per_1k_tokens = {
            "ollama": 0.0,      # Free local model
            "huggingface": 0.0,  # Free tier
            "groq": 0.001,      # Very cheap
            "openai": 0.002     # More expensive
        }
        
        rate = cost_per_1k_tokens.get(provider, 0.002)  # Default to OpenAI pricing
        return (token_count / 1000) * rate
    
    async def _find_cost_effective_provider(self, prompt: str, task_complexity: TaskComplexity) -> str:
        """Find the most cost-effective provider for the given task"""
        if self.config.prefer_free_providers:
            # Try free providers first
            free_providers = ["ollama", "huggingface"]
            for provider in free_providers:
                if await self._is_provider_available(provider):
                    return provider
        
        # Fall back to paid providers in order of cost
        paid_providers = ["groq", "openai"]
        for provider in paid_providers:
            if await self._is_provider_available(provider):
                return provider
        
        # Last resort: use configured preferred provider
        return self.config.preferred_ai_provider
    
    async def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available and performing well"""
        # Check provider performance history
        if provider in self._provider_performance:
            perf = self._provider_performance[provider]
            if perf['success_rate'] < 0.7:  # Less than 70% success rate
                return False
        
        # In production, you might ping the provider's health endpoint
        return True
    
    async def _try_fallback_providers(
        self, 
        prompt: str, 
        failed_provider: str,
        task_complexity: Optional[TaskComplexity] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Try fallback providers when primary provider fails"""
        
        fallback_providers = [p for p in self.config.fallback_providers if p != failed_provider]
        
        for provider in fallback_providers:
            try:
                self.logger.info(f"Trying fallback provider: {provider}")
                response = await self.ai_client.ask(
                    prompt=prompt,
                    provider=provider,
                    task_complexity=task_complexity or self.config.task_complexity,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                self.logger.info(f"Fallback provider {provider} succeeded")
                return response
                
            except Exception as e:
                self.logger.warning(f"Fallback provider {provider} failed: {e}")
                continue
        
        raise AgentError(self.config.name, "All fallback providers failed")
    
    def _update_provider_performance(self, provider: str, success: bool, response_time: float):
        """Update provider performance tracking"""
        if provider not in self._provider_performance:
            self._provider_performance[provider] = {
                'total_calls': 0,
                'successful_calls': 0,
                'total_response_time': 0.0,
                'success_rate': 0.0,
                'average_response_time': 0.0
            }
        
        perf = self._provider_performance[provider]
        perf['total_calls'] += 1
        perf['total_response_time'] += response_time
        
        if success:
            perf['successful_calls'] += 1
        
        perf['success_rate'] = perf['successful_calls'] / perf['total_calls']
        perf['average_response_time'] = perf['total_response_time'] / perf['total_calls']
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status and performance metrics"""
        performance_summary = self.performance_monitor.get_performance_summary()
        
        return {
            'agent_name': self.config.name,
            'health_status': self._health_status,
            'uptime_seconds': time.time() - self.performance_monitor.start_time,
            'performance': performance_summary,
            'provider_performance': self._provider_performance,
            'configuration': {
                'max_retries': self.config.max_retries,
                'rate_limit': self.config.rate_limit,
                'preferred_provider': self.config.preferred_ai_provider,
                'fallback_providers': self.config.fallback_providers,
                'cost_optimization': self.config.cost_optimization_enabled
            },
            'recent_costs': sum(self._operation_costs[-100:]) if self._operation_costs else 0.0  # Last 100 operations
        }
    
    async def execute_with_monitoring(
        self, 
        operation_func: Callable,
        operation_type: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute any operation with comprehensive monitoring"""
        operation_id = f"{self.config.name}_{operation_type}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            result = await operation_func(*args, **kwargs)
            
            # Record successful operation
            end_time = time.time()
            metrics = OperationMetrics(
                operation_id=operation_id,
                agent_name=self.config.name,
                operation_type=operation_type,
                status=OperationStatus.SUCCESS,
                start_time=start_time,
                end_time=end_time,
                execution_time=end_time - start_time,
                ai_provider_used="n/a",
                ai_calls_made=0,
                tokens_used=0,
                estimated_cost=0.0,
                confidence_score=getattr(result, 'confidence_score', 1.0)
            )
            
            self.performance_monitor.record_operation(metrics)
            return result
            
        except Exception as e:
            # Record failed operation
            end_time = time.time()
            metrics = OperationMetrics(
                operation_id=operation_id,
                agent_name=self.config.name,
                operation_type=operation_type,
                status=OperationStatus.FAILURE,
                start_time=start_time,
                end_time=end_time,
                execution_time=end_time - start_time,
                ai_provider_used="n/a",
                ai_calls_made=0,
                tokens_used=0,
                estimated_cost=0.0,
                confidence_score=0.0,
                error_message=str(e)
            )
            
            self.performance_monitor.record_operation(metrics)
            raise

    async def validate_response(
        self, 
        response: str, 
        expected_format: str = "json",
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Validate AI response format and content quality
        
        Args:
            response: AI response to validate
            expected_format: Expected response format (json, text, structured)
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Validation result with parsed content and confidence score
        """
        validation_result = {
            'is_valid': False,
            'parsed_content': None,
            'confidence_score': 0.0,
            'validation_errors': [],
            'format_detected': 'unknown'
        }
        
        try:
            if expected_format == "json":
                # Validate JSON format
                parsed_content = safe_json_parse(response)
                if parsed_content:
                    validation_result['is_valid'] = True
                    validation_result['parsed_content'] = parsed_content
                    validation_result['format_detected'] = 'json'
                    validation_result['confidence_score'] = 0.9  # High confidence for valid JSON
                else:
                    validation_result['validation_errors'].append("Invalid JSON format")
                    
            elif expected_format == "text":
                # Validate text content
                if response and len(response.strip()) > 10:
                    validation_result['is_valid'] = True
                    validation_result['parsed_content'] = response.strip()
                    validation_result['format_detected'] = 'text'
                    validation_result['confidence_score'] = 0.8
                else:
                    validation_result['validation_errors'].append("Text too short or empty")
                    
            elif expected_format == "structured":
                # Try to detect and validate structured content
                parsed_json = safe_json_parse(response)
                if parsed_json:
                    validation_result['is_valid'] = True
                    validation_result['parsed_content'] = parsed_json
                    validation_result['format_detected'] = 'json'
                    validation_result['confidence_score'] = 0.85
                elif response and len(response.strip()) > 50:
                    validation_result['is_valid'] = True
                    validation_result['parsed_content'] = response.strip()
                    validation_result['format_detected'] = 'text'
                    validation_result['confidence_score'] = 0.6
                else:
                    validation_result['validation_errors'].append("No valid structured content found")
            
            # Apply confidence threshold
            if validation_result['confidence_score'] < confidence_threshold:
                validation_result['is_valid'] = False
                validation_result['validation_errors'].append(
                    f"Confidence score {validation_result['confidence_score']:.2f} below threshold {confidence_threshold}"
                )
                
        except Exception as e:
            validation_result['validation_errors'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Response validation failed: {e}")
        
        return validation_result

    async def cache_operation_result(
        self, 
        cache_key: str, 
        result: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache operation result for future use
        
        Args:
            cache_key: Unique key for caching
            result: Result to cache
            ttl: Time to live override
            
        Returns:
            True if caching successful, False otherwise
        """
        try:
            # Use agent's default TTL if not specified
            actual_ttl = ttl or self.config.cache_ttl
            
            # Serialize result for caching
            if hasattr(result, 'to_dict'):
                serialized_result = result.to_dict()
            elif hasattr(result, '__dict__'):
                serialized_result = result.__dict__
            else:
                serialized_result = result
            
            cache_data = {
                'result': serialized_result,
                'cached_at': time.time(),
                'agent_name': self.config.name,
                'ttl': actual_ttl
            }
            
            # In production, this would integrate with Redis or similar
            # For now, we'll use a simple in-memory cache placeholder
            if not hasattr(self, '_cache'):
                self._cache = {}
            
            self._cache[cache_key] = cache_data
            
            # Cleanup expired cache entries periodically
            await self._cleanup_expired_cache()
            
            self.logger.debug(f"Cached result for key: {cache_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Caching failed for key {cache_key}: {e}")
            return False

    async def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve cached operation result
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Cached result if found and valid, None otherwise
        """
        try:
            if not hasattr(self, '_cache') or cache_key not in self._cache:
                return None
            
            cache_data = self._cache[cache_key]
            current_time = time.time()
            
            # Check if cache entry has expired
            if current_time - cache_data['cached_at'] > cache_data['ttl']:
                del self._cache[cache_key]
                return None
            
            self.logger.debug(f"Cache hit for key: {cache_key}")
            return cache_data['result']
            
        except Exception as e:
            self.logger.error(f"Cache retrieval failed for key {cache_key}: {e}")
            return None

    async def _cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        if not hasattr(self, '_cache'):
            return
        
        current_time = time.time()
        expired_keys = []
        
        for key, cache_data in self._cache.items():
            if current_time - cache_data['cached_at'] > cache_data['ttl']:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def log_operation(
        self, 
        operation_type: str, 
        details: Dict[str, Any],
        level: str = "info"
    ):
        """
        Log operation with structured details
        
        Args:
            operation_type: Type of operation being logged
            details: Operation details to log
            level: Log level (debug, info, warning, error)
        """
        log_entry = {
            'timestamp': time.time(),
            'agent_name': self.config.name,
            'operation_type': operation_type,
            'details': details
        }
        
        log_message = f"[{operation_type}] {json.dumps(details, default=str)}"
        
        if level == "debug":
            self.logger.debug(log_message)
        elif level == "info":
            self.logger.info(log_message)
        elif level == "warning":
            self.logger.warning(log_message)
        elif level == "error":
            self.logger.error(log_message)

    async def measure_performance(self, operation_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Measure performance of any operation
        
        Args:
            operation_func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Performance metrics including execution time, memory usage, etc.
        """
        import psutil
        import tracemalloc
        
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        start_memory = process.memory_info().rss
        start_time = time.time()
        start_cpu = process.cpu_percent()
        
        try:
            # Execute the operation
            result = await operation_func(*args, **kwargs)
            
            # Measure final state
            end_time = time.time()
            end_memory = process.memory_info().rss
            end_cpu = process.cpu_percent()
            
            # Get memory trace
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            performance_metrics = {
                'execution_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'peak_memory_usage': peak,
                'cpu_usage_delta': end_cpu - start_cpu,
                'operation_successful': True,
                'result_size': len(str(result)) if result else 0
            }
            
            await self.log_operation(
                "performance_measurement",
                performance_metrics,
                level="debug"
            )
            
            return {
                'result': result,
                'performance': performance_metrics
            }
            
        except Exception as e:
            tracemalloc.stop()
            end_time = time.time()
            
            performance_metrics = {
                'execution_time': end_time - start_time,
                'operation_successful': False,
                'error': str(e)
            }
            
            await self.log_operation(
                "performance_measurement_error",
                performance_metrics,
                level="error"
            )
            
            raise

    def calculate_confidence_score(
        self, 
        data_quality_indicators: Dict[str, Any],
        validation_results: Dict[str, Any],
        source_reliability: float = 0.8
    ) -> float:
        """
        Calculate confidence score based on multiple factors
        
        Args:
            data_quality_indicators: Indicators of data quality
            validation_results: Results from data validation
            source_reliability: Reliability of the data source
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.5  # Start with neutral confidence
        
        # Factor 1: Data completeness
        completeness = data_quality_indicators.get('completeness', 0.5)
        base_confidence += (completeness - 0.5) * 0.3
        
        # Factor 2: Data consistency
        consistency = data_quality_indicators.get('consistency', 0.5)
        base_confidence += (consistency - 0.5) * 0.2
        
        # Factor 3: Validation success
        if validation_results.get('is_valid', False):
            base_confidence += 0.2
        else:
            base_confidence -= 0.3
        
        # Factor 4: Source reliability
        base_confidence += (source_reliability - 0.5) * 0.2
        
        # Factor 5: Response format quality
        format_score = validation_results.get('confidence_score', 0.5)
        base_confidence += (format_score - 0.5) * 0.1
        
        # Ensure confidence is within valid range
        return max(0.0, min(1.0, base_confidence))

    async def optimize_ai_usage(
        self, 
        prompt: str, 
        task_complexity: TaskComplexity
    ) -> Dict[str, Any]:
        """
        Optimize AI usage based on cost, performance, and quality requirements
        
        Args:
            prompt: The prompt to optimize for
            task_complexity: Complexity of the task
            
        Returns:
            Optimization recommendations
        """
        optimization_result = {
            'recommended_provider': self.config.preferred_ai_provider,
            'estimated_cost': 0.0,
            'expected_quality': 0.8,
            'reasoning': []
        }
        
        try:
            # Analyze prompt complexity
            prompt_length = len(prompt.split())
            prompt_complexity = "simple" if prompt_length < 100 else "complex" if prompt_length < 500 else "very_complex"
            
            # Cost-based optimization
            if self.config.cost_optimization_enabled:
                if task_complexity == TaskComplexity.SIMPLE and prompt_complexity == "simple":
                    # Use free providers for simple tasks
                    optimization_result['recommended_provider'] = "ollama"
                    optimization_result['estimated_cost'] = 0.0
                    optimization_result['expected_quality'] = 0.7
                    optimization_result['reasoning'].append("Simple task routed to free provider")
                
                elif task_complexity == TaskComplexity.MEDIUM:
                    # Use HuggingFace for medium complexity
                    optimization_result['recommended_provider'] = "huggingface"
                    optimization_result['estimated_cost'] = 0.0
                    optimization_result['expected_quality'] = 0.8
                    optimization_result['reasoning'].append("Medium task routed to HuggingFace")
                
                elif task_complexity == TaskComplexity.COMPLEX:
                    # Check provider performance before expensive choice
                    if self._provider_performance.get("groq", {}).get('success_rate', 0.8) > 0.8:
                        optimization_result['recommended_provider'] = "groq"
                        optimization_result['estimated_cost'] = await self._estimate_operation_cost(prompt, "groq")
                        optimization_result['expected_quality'] = 0.85
                        optimization_result['reasoning'].append("Complex task routed to Groq for cost efficiency")
                    else:
                        optimization_result['recommended_provider'] = "openai"
                        optimization_result['estimated_cost'] = await self._estimate_operation_cost(prompt, "openai")
                        optimization_result['expected_quality'] = 0.9
                        optimization_result['reasoning'].append("Complex task routed to OpenAI for reliability")
            
            # Performance-based adjustments
            provider_perf = self._provider_performance.get(optimization_result['recommended_provider'], {})
            if provider_perf.get('success_rate', 1.0) < 0.7:
                # Switch to more reliable provider
                fallback_provider = "openai"
                optimization_result['recommended_provider'] = fallback_provider
                optimization_result['estimated_cost'] = await self._estimate_operation_cost(prompt, fallback_provider)
                optimization_result['reasoning'].append("Switched to reliable provider due to poor performance")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"AI usage optimization failed: {e}")
            return optimization_result

    async def handle_rate_limit_exceeded(self, provider: str, retry_after: Optional[int] = None):
        """
        Handle rate limit exceeded scenarios with intelligent backoff
        
        Args:
            provider: Provider that hit rate limit
            retry_after: Seconds to wait before retry (if provided by API)
        """
        # Calculate backoff time
        if retry_after:
            backoff_time = retry_after
        else:
            # Exponential backoff based on recent rate limit hits
            base_backoff = 60  # 1 minute base
            recent_rate_limits = sum(
                1 for m in self.performance_monitor.metrics_history[-10:]
                if m.status == OperationStatus.RATE_LIMITED and m.ai_provider_used == provider
            )
            backoff_time = min(base_backoff * (2 ** recent_rate_limits), 300)  # Max 5 minutes
        
        self.logger.warning(f"Rate limit exceeded for {provider}, backing off for {backoff_time}s")
        
        # Update provider performance
        self._update_provider_performance(provider, False, 0.0)
        
        # Log rate limit event
        await self.log_operation(
            "rate_limit_exceeded",
            {
                'provider': provider,
                'backoff_time': backoff_time,
                'retry_after': retry_after
            },
            level="warning"
        )
        
        await asyncio.sleep(backoff_time)

    def generate_operation_id(self, operation_type: str) -> str:
        """Generate unique operation ID"""
        import uuid
        timestamp = int(time.time() * 1000)
        unique_id = str(uuid.uuid4())[:8]
        return f"{self.config.name}_{operation_type}_{timestamp}_{unique_id}"

    async def batch_operations(
        self, 
        operations: List[Dict[str, Any]], 
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple operations in batches with concurrency control
        
        Args:
            operations: List of operations to execute
            max_concurrent: Maximum concurrent operations
            
        Returns:
            List of operation results
        """
        actual_max_concurrent = max_concurrent or self.config.max_concurrent_operations
        semaphore = asyncio.Semaphore(actual_max_concurrent)
        
        async def execute_single_operation(operation: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                operation_id = self.generate_operation_id("batch_operation")
                start_time = time.time()
                
                try:
                    # Execute the operation based on its type
                    if operation['type'] == 'ai_query':
                        result = await self.ask_ai(
                            prompt=operation['prompt'],
                            provider=operation.get('provider'),
                            task_complexity=operation.get('task_complexity')
                        )
                        
                    elif operation['type'] == 'function_call':
                        func = operation['function']
                        args = operation.get('args', [])
                        kwargs = operation.get('kwargs', {})
                        result = await func(*args, **kwargs)
                        
                    else:
                        raise ValueError(f"Unknown operation type: {operation['type']}")
                    
                    return {
                        'operation_id': operation_id,
                        'status': 'success',
                        'result': result,
                        'execution_time': time.time() - start_time
                    }
                    
                except Exception as e:
                    return {
                        'operation_id': operation_id,
                        'status': 'error',
                        'error': str(e),
                        'execution_time': time.time() - start_time
                    }
        
        # Execute all operations concurrently
        tasks = [execute_single_operation(op) for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in the gather
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'operation_id': f"failed_{i}",
                    'status': 'error',
                    'error': str(result),
                    'execution_time': 0.0
                })
            else:
                processed_results.append(result)
        
        await self.log_operation(
            "batch_operations_completed",
            {
                'total_operations': len(operations),
                'successful_operations': sum(1 for r in processed_results if r['status'] == 'success'),
                'max_concurrent': actual_max_concurrent
            }
        )
        
        return processed_results

    def export_metrics(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export agent metrics in specified format
        
        Args:
            format: Export format (json, prometheus, csv)
            
        Returns:
            Formatted metrics data
        """
        metrics_data = {
            'agent_info': {
                'name': self.config.name,
                'uptime': time.time() - self.performance_monitor.start_time,
                'status': self._health_status
            },
            'performance': self.performance_monitor.get_performance_summary(),
            'provider_performance': self._provider_performance,
            'configuration': {
                'max_retries': self.config.max_retries,
                'rate_limit': self.config.rate_limit,
                'cache_ttl': self.config.cache_ttl,
                'cost_optimization_enabled': self.config.cost_optimization_enabled
            }
        }
        
        if format == "json":
            return metrics_data
        elif format == "prometheus":
            # Convert to Prometheus format
            prometheus_metrics = []
            for key, value in metrics_data['performance'].items():
                if isinstance(value, (int, float)):
                    prometheus_metrics.append(f"agent_{key}{{agent=\"{self.config.name}\"}} {value}")
            return "\n".join(prometheus_metrics)
        elif format == "csv":
            # Convert to CSV format (simplified)
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['metric', 'value'])
            
            def flatten_dict(d, prefix=''):
                items = []
                for k, v in d.items():
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, f"{prefix}{k}_"))
                    else:
                        items.append((f"{prefix}{k}", v))
                return items
            
            for metric, value in flatten_dict(metrics_data):
                writer.writerow([metric, value])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def cleanup_resources(self):
        """Comprehensive resource cleanup"""
        try:
            # Clear cache
            if hasattr(self, '_cache'):
                self._cache.clear()
            
            # Export final metrics
            final_metrics = self.export_metrics()
            
            # Log final performance summary
            await self.log_operation(
                "agent_shutdown",
                {
                    'final_metrics': final_metrics,
                    'total_uptime': time.time() - self.performance_monitor.start_time
                }
            )
            
            # Call agent-specific cleanup
            await self._cleanup_agent()
            
        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")


# Utility functions for common agent operations

async def create_agent_with_monitoring(
    agent_class, 
    config: AgentConfig, 
    ai_client: AIAsyncClient
) -> BaseAgent:
    """
    Create agent instance with comprehensive monitoring setup
    
    Args:
        agent_class: Agent class to instantiate
        config: Agent configuration
        ai_client: AI client instance
        
    Returns:
        Configured agent instance
    """
    agent = agent_class(config, ai_client)
    
    # Setup monitoring
    logger.info(f"Initializing agent: {config.name}")
    
    return agent


def validate_agent_config(config: AgentConfig) -> List[str]:
    """
    Validate agent configuration and return any issues
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    if not config.name:
        issues.append("Agent name is required")
    
    if config.max_retries < 0:
        issues.append("Max retries must be non-negative")
    
    if config.rate_limit <= 0:
        issues.append("Rate limit must be positive")
    
    if config.cache_ttl < 0:
        issues.append("Cache TTL must be non-negative")
    
    if config.min_confidence_score < 0 or config.min_confidence_score > 1:
        issues.append("Min confidence score must be between 0 and 1")
    
    if config.max_concurrent_operations <= 0:
        issues.append("Max concurrent operations must be positive")
    
    return issues


# Export main classes and functions
__all__ = [
    'BaseAgent',
    'AgentConfig', 
    'AgentError',
    'OperationStatus',
    'OperationMetrics',
    'PerformanceMonitor',
    'safe_json_parse',
    'retry_with_backoff',
    'rate_limiter',
    'create_agent_with_monitoring',
    'validate_agent_config'
]