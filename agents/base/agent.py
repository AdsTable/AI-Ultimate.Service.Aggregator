# agents/base/agent.py
"""
Base agent class with integrated AIAsyncClient support
"""

import asyncio
import logging
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from collections import deque

from services.ai_async_client import AIAsyncClient, TaskComplexity

logger = logging.getLogger(__name__)

class AgentError(Exception):
    """Base exception for agent operations"""
    
    def __init__(self, agent_name: str, message: str, error_code: Optional[str] = None):
        self.agent_name = agent_name
        self.error_code = error_code
        super().__init__(f"[{agent_name}] {message}")

@dataclass
class AgentMetrics:
    """Performance metrics for agents"""
    
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    avg_response_time: float = 0.0
    last_operation_time: Optional[datetime] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_success(self, response_time: float):
        """Record successful operation"""
        self.total_operations += 1
        self.successful_operations += 1
        self.last_operation_time = datetime.now()
        self.response_times.append(response_time)
        self.avg_response_time = sum(self.response_times) / len(self.response_times)
    
    def record_failure(self):
        """Record failed operation"""
        self.total_operations += 1
        self.failed_operations += 1
        self.last_operation_time = datetime.now()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations

@dataclass
class AgentConfig:
    """Configuration for agents"""
    
    name: str
    max_retries: int = 3
    timeout: int = 30
    rate_limit: int = 60  # operations per minute
    preferred_ai_provider: Optional[str] = None
    task_complexity: TaskComplexity = TaskComplexity.MEDIUM
    cache_ttl: int = 3600
    debug: bool = False

class BaseAgent(ABC):
    """
    Base class for all AI agents with integrated AIAsyncClient support
    
    Features:
    - Automatic integration with existing AIAsyncClient
    - Performance metrics tracking
    - Rate limiting and retry logic
    - Error handling and logging
    - Configuration management
    """
    
    def __init__(self, config: AgentConfig, ai_client: AIAsyncClient):
        self.config = config
        self.ai_client = ai_client
        self.metrics = AgentMetrics()
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self._last_operation_time = 0.0
        
        # Setup logging level
        if config.debug:
            self.logger.setLevel(logging.DEBUG)
        
        self.logger.info(f"✅ Initialized {config.name} agent")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize agent resources"""
        try:
            await self._setup_agent()
            self.logger.info(f"🚀 Agent {self.config.name} initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize {self.config.name}: {e}")
            raise AgentError(self.config.name, f"Initialization failed: {e}")
    
    async def cleanup(self):
        """Cleanup agent resources"""
        try:
            await self._cleanup_agent()
            self.logger.info(f"🔄 Agent {self.config.name} cleanup completed")
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup warning for {self.config.name}: {e}")
    
    async def _setup_agent(self):
        """Agent-specific setup logic (override in subclasses)"""
        pass
    
    async def _cleanup_agent(self):
        """Agent-specific cleanup logic (override in subclasses)"""
        pass
    
    async def _apply_rate_limit(self):
        """Apply rate limiting between operations"""
        now = time.time()
        time_since_last = now - self._last_operation_time
        min_interval = 60.0 / self.config.rate_limit
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self._last_operation_time = time.time()
    
    async def ask_ai(
        self, 
        prompt: str, 
        provider: Optional[str] = None,
        task_complexity: Optional[TaskComplexity] = None,
        **kwargs
    ) -> str:
        """
        Convenient method to ask AI with agent-specific configuration
        """
        await self._apply_rate_limit()
        
        # Use agent configuration as defaults
        actual_provider = provider or self.config.preferred_ai_provider
        actual_complexity = task_complexity or self.config.task_complexity
        
        try:
            start_time = time.time()
            
            response = await self.ai_client.ask(
                prompt=prompt,
                provider=actual_provider,
                task_complexity=actual_complexity,
                **kwargs
            )
            
            response_time = (time.time() - start_time) * 1000
            self.metrics.record_success(response_time)
            
            self.logger.debug(f"✅ AI response received ({response_time:.0f}ms)")
            return response
            
        except Exception as e:
            self.metrics.record_failure()
            self.logger.error(f"❌ AI request failed: {e}")
            raise AgentError(
                self.config.name, 
                f"AI request failed: {e}",
                error_code="ai_request_failed"
            )
    
    async def batch_ask_ai(
        self, 
        prompts: List[str], 
        max_concurrency: int = 3,
        **kwargs
    ) -> List[str]:
        """
        Batch AI requests with concurrency control
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_prompt(prompt: str) -> str:
            async with semaphore:
                return await self.ask_ai(prompt, **kwargs)
        
        tasks = [process_prompt(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return {
            "agent_name": self.config.name,
            "total_operations": self.metrics.total_operations,
            "success_rate": f"{self.metrics.success_rate * 100:.1f}%",
            "avg_response_time": f"{self.metrics.avg_response_time:.0f}ms",
            "last_operation": self.metrics.last_operation_time.isoformat() if self.metrics.last_operation_time else None,
            "config": {
                "max_retries": self.config.max_retries,
                "rate_limit": self.config.rate_limit,
                "preferred_provider": self.config.preferred_ai_provider,
                "task_complexity": self.config.task_complexity.value
            }
        }
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """
        Main execution method for the agent
        Must be implemented by subclasses
        """
        pass