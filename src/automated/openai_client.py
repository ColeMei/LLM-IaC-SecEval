"""
OpenAI API client for cloud-based LLM inference
"""
from openai import OpenAI
import time
from typing import Dict, Any, Optional
from .model_client import ModelClient, ModelResponse
from .config import config

class OpenAIClient(ModelClient):
    """Client for interacting with OpenAI API"""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name_str = model_name or config.openai_default_model
        self.api_key = api_key or config.openai_api_key
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it in config or pass as parameter.")
        
        # Initialize OpenAI client (new v1.0+ API)
        self.client = OpenAI(api_key=self.api_key)
        
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate response using OpenAI API
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse object with generated content
        """
        start_time = time.time()
        
        try:
            # Prepare chat messages format for OpenAI
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Map parameters to OpenAI format
            openai_params = {
                "model": self.model_name_str,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 512),
                "top_p": kwargs.get("top_p", 0.9),
                "frequency_penalty": kwargs.get("frequency_penalty", 0),
                "presence_penalty": kwargs.get("presence_penalty", 0),
            }
            
            # Make API call (new v1.0+ API)
            response = self.client.chat.completions.create(**openai_params)
            end_time = time.time()
            
            # Extract response data
            choice = response.choices[0]
            usage = response.usage
            
            return ModelResponse(
                content=choice.message.content,
                model_name=self.model_name_str,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                response_time=end_time - start_time,
                metadata={
                    "finish_reason": choice.finish_reason,
                    "response_id": response.id,
                    "created": response.created,
                    "object": response.object,
                }
            )
            
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower():
                raise ConnectionError(f"OpenAI authentication failed: {e}")
            elif "rate_limit" in error_msg.lower() or "429" in error_msg:
                raise ConnectionError(f"OpenAI rate limit exceeded: {e}")
            else:
                raise ConnectionError(f"OpenAI API error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during OpenAI generation: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            # Try to list models to check API access (new v1.0+ API)
            self.client.models.list()
            return True
        except Exception:
            return False
    
    @property
    def model_name(self) -> str:
        """Get the model identifier"""
        return self.model_name_str
    
    def list_models(self) -> list:
        """List all available OpenAI models"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception:
            return []
    
    def check_model_availability(self) -> bool:
        """Check if the specific model is available"""
        try:
            available_models = self.list_models()
            return self.model_name_str in available_models
        except Exception:
            return False