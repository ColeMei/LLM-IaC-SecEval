"""
Abstract model client interface for extensibility
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelResponse:
    """Standardized response from any LLM"""
    content: str
    model_name: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    response_time: Optional[float] = None
    metadata: Dict[str, Any] = None

class ModelClient(ABC):
    """Abstract base class for all LLM clients"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from the model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available/reachable"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model identifier"""
        pass