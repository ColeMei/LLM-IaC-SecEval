"""
Backward-compatible wrapper for External Prompt System

This module provides backward compatibility for existing code while using the new external prompt system.
It maintains the same API as the original prompt_templates.py but loads prompts from external files.
"""

from typing import Dict, Optional
from .prompt_loader import ExternalPromptLoader, SecuritySmell, PromptStyle, PromptVersion


class SecuritySmellPrompts:
    """Backward-compatible container for security smell definitions and prompt templates."""
    
    def __init__(self, prompt_version: str = PromptVersion.CURRENT.value):
        """Initialize with specified prompt version."""
        self.loader = ExternalPromptLoader(prompt_version)
    
    @property
    def DEFINITIONS(self) -> Dict[SecuritySmell, str]:
        """Get all definitions as a dictionary for backward compatibility."""
        return {smell: self.loader.get_definition(smell) for smell in SecuritySmell}
    
    @classmethod
    def get_definition_based_prompt(cls, prompt_version: str = PromptVersion.CURRENT.value) -> str:
        """Get the definition-based prompt template."""
        loader = ExternalPromptLoader(prompt_version)
        return loader.get_base_prompt(PromptStyle.DEFINITION_BASED)
    
    @classmethod
    def get_static_analysis_rules_prompt(cls, prompt_version: str = PromptVersion.CURRENT.value) -> str:
        """Get the static analysis rules-based prompt template."""
        loader = ExternalPromptLoader(prompt_version)
        return loader.get_base_prompt(PromptStyle.STATIC_ANALYSIS_RULES)
    
    @classmethod
    def get_base_prompt(cls, style: PromptStyle = PromptStyle.DEFINITION_BASED, prompt_version: str = PromptVersion.CURRENT.value) -> str:
        """Get the base prompt template for the specified style."""
        loader = ExternalPromptLoader(prompt_version)
        return loader.get_base_prompt(style)
    
    @classmethod
    def create_prompt(
        cls, 
        smell: SecuritySmell, 
        code_snippet: str, 
        style: PromptStyle = PromptStyle.DEFINITION_BASED,
        prompt_version: str = PromptVersion.CURRENT.value
    ) -> str:
        """Create a complete prompt for a specific security smell and code snippet."""
        loader = ExternalPromptLoader(prompt_version)
        return loader.create_prompt(smell, code_snippet, style)
    
    @classmethod
    def get_smell_from_string(cls, smell_name: str) -> Optional[SecuritySmell]:
        """Convert string smell name to SecuritySmell enum."""
        return ExternalPromptLoader.get_smell_from_string(smell_name)
    
    @classmethod
    def get_prompt_style_from_string(cls, style_name: str) -> Optional[PromptStyle]:
        """Convert string style name to PromptStyle enum."""
        return ExternalPromptLoader.get_prompt_style_from_string(style_name)
    
    @classmethod
    def get_validation_prompt(cls, smell: SecuritySmell, style: PromptStyle = PromptStyle.DEFINITION_BASED, prompt_version: str = PromptVersion.CURRENT.value) -> str:
        """Get a validation prompt to test LLM understanding of the smell definition."""
        loader = ExternalPromptLoader(prompt_version)
        return loader.get_validation_prompt(smell, style)


# For backward compatibility with existing imports
from .prompt_loader import SecuritySmell, PromptStyle

# Add PromptVersion for new functionality
__all__ = [
    'SecuritySmellPrompts',
    'SecuritySmell', 
    'PromptStyle',
    'PromptVersion'
]

