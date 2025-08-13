"""
Anthropic Claude API client for cloud-based LLM inference
"""
from anthropic import Anthropic
import time
from typing import Dict, Any, Optional
from .model_client import ModelClient, ModelResponse
from .config import config

class ClaudeClient(ModelClient):
    """Client for interacting with Anthropic Claude API"""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name_str = model_name or config.claude_default_model
        self.api_key = api_key or config.claude_api_key
        
        if not self.api_key:
            raise ValueError("Claude API key is required. Set it in config or pass as parameter.")
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=self.api_key)
        
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate response using Claude API
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse object with generated content
        """
        start_time = time.time()
        
        try:
            # Prepare message format for Claude
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Map parameters to Claude format
            claude_params = {
                "model": self.model_name_str,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 512),
                "top_p": kwargs.get("top_p", 0.9),
            }
            
            # Make API call
            response = self.client.messages.create(**claude_params)
            end_time = time.time()
            
            # Extract response data
            content = ""
            if response.content:
                # Claude returns content as a list of content blocks
                content = "".join([block.text for block in response.content if hasattr(block, 'text')])
            
            return ModelResponse(
                content=content,
                model_name=self.model_name_str,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                response_time=end_time - start_time,
                metadata={
                    "id": response.id,
                    "type": response.type,
                    "role": response.role,
                    "stop_reason": response.stop_reason,
                    "stop_sequence": response.stop_sequence,
                }
            )
            
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                raise ConnectionError(f"Claude authentication failed: {e}")
            elif "rate_limit" in error_msg.lower() or "429" in error_msg:
                raise ConnectionError(f"Claude rate limit exceeded: {e}")
            elif "overloaded" in error_msg.lower() or "529" in error_msg:
                raise ConnectionError(f"Claude API overloaded: {e}")
            else:
                raise ConnectionError(f"Claude API error: {e}")
    
    def is_available(self) -> bool:
        """Check if Claude API is accessible"""
        try:
            # Make a minimal API call to check access
            test_response = self.client.messages.create(
                model=self.model_name_str,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )
            return True
        except Exception:
            return False
    
    @property
    def model_name(self) -> str:
        """Get the model identifier"""
        return self.model_name_str
    
    def list_models(self) -> list:
        """
        List available Claude models
        Note: Anthropic doesn't provide a models endpoint, so we return common models
        """
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022", 
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
    
    def check_model_availability(self) -> bool:
        """Check if the specific model is available"""
        available_models = self.list_models()
        return self.model_name_str in available_models
