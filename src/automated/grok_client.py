"""
Grok (X.AI) API client for cloud-based LLM inference
"""
import requests
import json
import time
from typing import Dict, Any, Optional
from .model_client import ModelClient, ModelResponse
from .config import config

class GrokClient(ModelClient):
    """Client for interacting with Grok (X.AI) API"""
    
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        self.model_name_str = model_name or config.grok_default_model
        self.api_key = api_key or config.grok_api_key
        self.base_url = base_url or config.grok_base_url
        
        if not self.api_key:
            raise ValueError("Grok API key is required. Set it in config or pass as parameter.")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate response using Grok API
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse object with generated content
        """
        start_time = time.time()
        
        try:
            # Prepare chat messages format (OpenAI-compatible)
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Prepare request payload
            payload = {
                "model": self.model_name_str,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 512),
                "top_p": kwargs.get("top_p", 0.9),
                "frequency_penalty": kwargs.get("frequency_penalty", 0),
                "presence_penalty": kwargs.get("presence_penalty", 0),
                "stream": False
            }
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=kwargs.get("timeout", 120)
            )
            response.raise_for_status()
            
            result = response.json()
            end_time = time.time()
            
            # Extract response data
            choice = result["choices"][0]
            usage = result.get("usage", {})
            
            return ModelResponse(
                content=choice["message"]["content"],
                model_name=self.model_name_str,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                response_time=end_time - start_time,
                metadata={
                    "finish_reason": choice.get("finish_reason"),
                    "response_id": result.get("id"),
                    "created": result.get("created"),
                    "object": result.get("object"),
                    "model": result.get("model"),
                }
            )
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 401:
                    raise ConnectionError(f"Grok authentication failed: {e}")
                elif e.response.status_code == 429:
                    raise ConnectionError(f"Grok rate limit exceeded: {e}")
                else:
                    raise ConnectionError(f"Grok API error: {e}")
            else:
                raise ConnectionError(f"Failed to connect to Grok API: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from Grok: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during Grok generation: {e}")
    
    def is_available(self) -> bool:
        """Check if Grok API is accessible"""
        try:
            # Try to list models to check API access
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception:
            return False
    
    @property
    def model_name(self) -> str:
        """Get the model identifier"""
        return self.model_name_str
    
    def list_models(self) -> list:
        """List all available Grok models"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            models = response.json()
            return [model["id"] for model in models.get("data", [])]
        except Exception:
            return []
    
    def check_model_availability(self) -> bool:
        """Check if the specific model is available"""
        try:
            available_models = self.list_models()
            return self.model_name_str in available_models
        except Exception:
            return False