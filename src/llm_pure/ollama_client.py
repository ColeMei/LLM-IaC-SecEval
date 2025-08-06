"""
Ollama API client for local LLM inference
"""
import requests
import json
import time
from typing import Dict, Any, Optional
from .model_client import ModelClient, ModelResponse
from .config import config

class OllamaClient(ModelClient):
    """Client for interacting with local Ollama server"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name_str = model_name or config.default_model
        self.base_url = base_url or config.ollama_base_url
        self.api_url = f"{self.base_url}/api/generate"
        
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate response using Ollama API
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse object with generated content
        """
        start_time = time.time()
        
        # Prepare request payload
        payload = {
            "model": self.model_name_str,
            "prompt": prompt,
            "stream": False,  # Get complete response
            "options": {
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40),
                "num_predict": kwargs.get("max_tokens", 512),
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=kwargs.get("timeout", 120)
            )
            response.raise_for_status()
            
            result = response.json()
            end_time = time.time()
            
            return ModelResponse(
                content=result.get("response", ""),
                model_name=self.model_name_str,
                prompt_tokens=result.get("prompt_eval_count"),
                completion_tokens=result.get("eval_count"),
                total_tokens=result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
                response_time=end_time - start_time,
                metadata={
                    "done": result.get("done", False),
                    "context": result.get("context", []),
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration"),
                    "prompt_eval_duration": result.get("prompt_eval_duration"),
                    "eval_duration": result.get("eval_duration"),
                }
            )
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama server: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from Ollama: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during generation: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama server is running and model is available"""
        try:
            # Check server status
            health_url = f"{self.base_url}/api/tags"
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
            
            # Check if specific model is available
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            return any(self.model_name_str in name for name in model_names)
            
        except Exception:
            return False
    
    @property
    def model_name(self) -> str:
        """Get the model identifier"""
        return self.model_name_str
    
    def pull_model(self) -> bool:
        """
        Pull the model if not available locally
        
        Returns:
            True if successful, False otherwise
        """
        try:
            pull_url = f"{self.base_url}/api/pull"
            payload = {"name": self.model_name_str}
            
            response = requests.post(pull_url, json=payload, timeout=300)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            print(f"Failed to pull model {self.model_name_str}: {e}")
            return False
    
    def list_models(self) -> list:
        """List all available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            return [model.get("name", "") for model in models]
            
        except Exception:
            return []