"""
Backward-compatible wrapper for External Prompt System

This module provides backward compatibility for existing code while using the new external prompt system.
It maintains the same API as the original prompt_templates.py but loads prompts from external files.
"""

from typing import Dict, Optional
from .prompt_loader import ExternalPromptLoader, SecuritySmell


class SecuritySmellPrompts:
    """Backward-compatible container for security smell definitions and prompt templates."""
    
    def __init__(self, prompt_template: str = "definition_based_conservative"):
        """Initialize with specified prompt template."""
        self.loader = ExternalPromptLoader(prompt_template)
    
    @property
    def DEFINITIONS(self) -> Dict[SecuritySmell, str]:
        """Get all definitions as a dictionary for backward compatibility."""
        return {smell: self.loader.get_definition(smell) for smell in SecuritySmell}
    
    @classmethod
    def create_prompt(cls, smell: SecuritySmell, code_snippet: str, prompt_template: str = "definition_based_conservative", iac_tech: Optional[str] = None) -> str:
        """Create a complete prompt for a specific security smell and code snippet.
        Optionally include the IaC technology name to specialize the prompt text.
        """
        loader = ExternalPromptLoader(prompt_template)
        return loader.create_prompt(smell, code_snippet, iac_tech)
    
    @classmethod
    def get_base_prompt(cls, prompt_template: str = "definition_based_conservative") -> str:
        """Get the base prompt template."""
        loader = ExternalPromptLoader(prompt_template)
        return loader.get_base_prompt()
    
    @classmethod
    def get_available_templates(cls) -> list:
        """Get list of available prompt templates."""
        return ExternalPromptLoader.get_available_templates()
    
    @classmethod
    def get_smell_from_string(cls, smell_name: str) -> Optional[SecuritySmell]:
        """Convert string smell name to SecuritySmell enum."""
        return ExternalPromptLoader.get_smell_from_string(smell_name)
    



# For backward compatibility with existing imports
from .prompt_loader import SecuritySmell

__all__ = [
    'SecuritySmellPrompts',
    'SecuritySmell'
]

