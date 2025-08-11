# LLM-IaC-SecEval Automated Pipeline
__version__ = "1.0.0"

# Import core components
from .model_client import ModelClient, ModelResponse
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .config import config
from .pipeline import LLMIaCPipeline
from .evaluator import Evaluator

# Client factory function
def create_client(client_type: str, **kwargs):
    """
    Factory function to create different types of LLM clients
    
    Args:
        client_type: Type of client ('ollama', 'openai')
        **kwargs: Additional parameters for client initialization
        
    Returns:
        ModelClient instance
        
    Raises:
        ValueError: If client_type is not supported
    """
    client_type = client_type.lower()
    
    if client_type == 'ollama':
        return OllamaClient(**kwargs)
    elif client_type == 'openai':
        return OpenAIClient(**kwargs)
    else:
        raise ValueError(f"Unsupported client type: {client_type}. Supported types: ollama, openai")

# Available client types and prompt styles
SUPPORTED_CLIENTS = ['ollama', 'openai']
SUPPORTED_PROMPT_STYLES = ['definition_based', 'static_analysis_rules']

__all__ = [
    'ModelClient', 'ModelResponse',
    'OllamaClient', 'OpenAIClient',
    'config', 'LLMIaCPipeline', 'Evaluator',
    'create_client', 'SUPPORTED_CLIENTS'
]