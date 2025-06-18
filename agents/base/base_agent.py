# agents/base/base_agent.py
"""
Base agent class with common functionality for all AI agents
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from collections import deque

from services.ai_async_client import AIAsyncClient


@dataclass
class AgentMetrics:
    """Performance metrics for agent operations"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_latency_ms: float = 0.0
    last_operation_time: Optional[datetime] = None
    operation_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_operation(self, success: bool, latency_ms: float):
        """Record an operation result"""
        self.total_operations += 1
        self.last_operation_time = datetime.now()
        
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
            
        self.operation_history.append({
            'success': success,
            'latency_ms': latency_ms,
            'timestamp': self.last_operation_time
        })
        
        # Update average latency
        recent_latencies = [op['latency_ms'] for op in self.operation_history if op['success']]
        if recent_latencies:
            self.average_latency_ms = sum(recent_latencies) / len(recent_latencies)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations


class AgentError(Exception):
    """Base exception for agent-related errors"""
    def __init__(self, agent_name: str, message: str, error_code: Optional[str] = None):
        self.agent_name = agent_name
        self.error_code = error_code
        super().__init__(f"[{agent_name}] {message}")


class BaseAgent(ABC):
    """
    Base class for all AI agents with common functionality
    
    Features:
    - Integration with AIAsyncClient for cost-optimized AI usage
    - Performance metrics tracking
    - Error handling and retry logic
    - Async/await support
    - Logging and monitoring
    """
    
    def __init__(self, ai_client: AIAsyncClient, agent_name: str):
        self.ai_client = ai_client
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"agents.{agent_name}")
        self.metrics = AgentMetrics()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the agent"""
        try:
            await self._setup()
            self._initialized = True
            self.logger.info(f"✅ {self.agent_name} agent initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize {self.agent_name} agent: {e}")
            raise AgentError(self.agent_name, f"Initialization failed: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        try:
            await self._teardown()
            self.logger.info(f"🔄 {self.agent_name} agent cleanup completed")
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup warning for {self.agent_name}: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self._initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    @abstractmethod
    async def _setup(self) -> None:
        """Agent-specific setup logic (override in subclasses)"""
        pass
    
    @abstractmethod
    async def _teardown(self) -> None:
        """Agent-specific cleanup logic (override in subclasses)"""
        pass
    
    async def ask_ai(
        self, 
        prompt: str, 
        provider: Optional[str] = None,
        task_complexity: str = "medium",
        **kwargs
    ) -> str:
        """
        Ask AI with automatic error handling and metrics
        
        Args:
            prompt: AI prompt
            provider: Preferred AI provider
            task_complexity: Task complexity for provider selection
            **kwargs: Additional parameters for AI client
            
        Returns:
            AI response text
        """
        start_time = time.time()
        
        try:
            response = await self.ai_client.ask(
                prompt=prompt,
                provider=provider,
                task_complexity=task_complexity,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation(True, latency_ms)
            
            self.logger.debug(f"AI request successful: {latency_ms:.0f}ms")
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation(False, latency_ms)
            
            self.logger.error(f"AI request failed: {e}")
            raise AgentError(self.agent_name, f"AI request failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "agent_name": self.agent_name,
            "total_operations": self.metrics.total_operations,
            "success_rate": f"{self.metrics.success_rate * 100:.1f}%",
            "average_latency_ms": f"{self.metrics.average_latency_ms:.0f}ms",
            "last_operation": self.metrics.last_operation_time.isoformat() if self.metrics.last_operation_time else None,
            "initialized": self._initialized
        }