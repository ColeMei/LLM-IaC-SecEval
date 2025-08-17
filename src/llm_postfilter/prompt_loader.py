"""
External Prompt Loader for LLM Post-Filter

This module loads prompts and definitions from external files using filename suffixes.
Supports multiple prompt versions for easy experimentation.
"""

import yaml
from pathlib import Path
from typing import Dict, Optional
from enum import Enum


class SecuritySmell(Enum):
    """Enumeration of security smell categories."""
    HARD_CODED_SECRET = "Hard-coded secret"
    SUSPICIOUS_COMMENT = "Suspicious comment"
    WEAK_CRYPTOGRAPHY = "Use of weak cryptography algorithms"


class PromptStyle(Enum):
    """Enumeration of available prompt styles."""
    DEFINITION_BASED = "definition_based"
    STATIC_ANALYSIS_RULES = "static_analysis_rules"


class PromptVersion(Enum):
    """Enumeration of prompt versions."""
    CURRENT = "current"
    CONSERVATIVE = "conservative"


class ExternalPromptLoader:
    """Loads prompts and definitions from external files using filename suffixes."""
    
    def __init__(self, prompt_version: str = PromptVersion.CURRENT.value):
        """Initialize with specified prompt version."""
        self.prompt_version = prompt_version
        self.project_root = Path(__file__).parent.parent.parent
        self.prompts_dir = self.project_root / "src" / "prompts" / "llm_postfilter"
        
        # Validate prompt version
        if prompt_version not in [v.value for v in PromptVersion]:
            raise ValueError(f"Invalid prompt version: {prompt_version}. Must be: {[v.value for v in PromptVersion]}")
        
        # Load definitions once during initialization
        self._definitions = self._load_smell_definitions()
    
    def _get_filename(self, base_name: str) -> Path:
        """Get filename with version suffix."""
        return self.prompts_dir / f"{base_name}_{self.prompt_version}.txt"
    
    def _get_yaml_filename(self, base_name: str) -> Path:
        """Get YAML filename with version suffix."""
        return self.prompts_dir / f"{base_name}_{self.prompt_version}.yaml"
    
    def _load_smell_definitions(self) -> Dict[SecuritySmell, str]:
        """Load security smell definitions from YAML file."""
        definitions_file = self._get_yaml_filename("smell_definitions")
        
        if not definitions_file.exists():
            raise FileNotFoundError(f"Definitions file not found: {definitions_file}")
        
        with open(definitions_file, 'r', encoding='utf-8') as f:
            definitions_dict = yaml.safe_load(f)
        
        # Convert to SecuritySmell enum mapping
        definitions = {}
        for smell in SecuritySmell:
            if smell.value in definitions_dict:
                definitions[smell] = definitions_dict[smell.value].strip()
            else:
                raise KeyError(f"Definition for '{smell.value}' not found in {definitions_file}")
        
        return definitions
    
    def _load_prompt_template(self, style: PromptStyle) -> str:
        """Load prompt template from file."""
        if style == PromptStyle.DEFINITION_BASED:
            template_file = self._get_filename("definition_based_prompt")
        else:  # STATIC_ANALYSIS_RULES
            template_file = self._get_filename("static_analysis_rules_prompt")
        
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")
        
        return template_file.read_text(encoding='utf-8')
    
    def get_definition(self, smell: SecuritySmell) -> str:
        """Get definition for a specific security smell."""
        return self._definitions[smell]
    
    def create_prompt(
        self, 
        smell: SecuritySmell, 
        code_snippet: str, 
        style: PromptStyle = PromptStyle.DEFINITION_BASED
    ) -> str:
        """Create a complete prompt for a specific security smell and code snippet."""
        template = self._load_prompt_template(style)
        
        if style == PromptStyle.DEFINITION_BASED:
            return template.format(
                smell_definition=self.get_definition(smell),
                smell_name=smell.value,
                code_snippet=code_snippet
            )
        else:  # STATIC_ANALYSIS_RULES
            return template.format(
                smell_name=smell.value,
                code_snippet=code_snippet
            )
    
    def get_base_prompt(self, style: PromptStyle = PromptStyle.DEFINITION_BASED) -> str:
        """Get the base prompt template for the specified style."""
        return self._load_prompt_template(style)
    
    @classmethod
    def get_smell_from_string(cls, smell_name: str) -> Optional[SecuritySmell]:
        """Convert string smell name to SecuritySmell enum."""
        for smell in SecuritySmell:
            if smell.value == smell_name:
                return smell
        return None
    
    @classmethod
    def get_prompt_style_from_string(cls, style_name: str) -> Optional[PromptStyle]:
        """Convert string style name to PromptStyle enum."""
        for style in PromptStyle:
            if style.value == style_name:
                return style
        return None
    
    @classmethod
    def get_available_versions(cls) -> list:
        """Get list of available prompt versions by scanning filenames."""
        prompts_dir = Path(__file__).parent.parent.parent / "src" / "prompts" / "llm_postfilter"
        versions = set()
        
        for file in prompts_dir.glob("*_*.txt"):
            # Extract version from filename (e.g., "prompt_current.txt" -> "current")
            parts = file.stem.split('_')
            if len(parts) >= 2:
                version = parts[-1]
                versions.add(version)
        
        return sorted(list(versions))
    
    def get_validation_prompt(self, smell: SecuritySmell, style: PromptStyle = PromptStyle.DEFINITION_BASED) -> str:
        """Get a validation prompt to test LLM understanding of the smell definition."""
        if style == PromptStyle.STATIC_ANALYSIS_RULES:
            return f"""Please confirm your understanding of the static analysis rules for "{smell.value}":

Based on the rule-based detection approach, do you understand how to identify {smell.value} using logical conditions and keyword matching functions?

Please respond with "YES" if you understand, or ask for clarification if needed."""
        else:
            definition = self.get_definition(smell)
            return f"""Please confirm your understanding of "{smell.value}":

{definition}

Do you understand this definition? Please respond with "YES" if you understand, or ask for clarification if needed."""


def main():
    """Test the external prompt loader with both versions."""
    test_code = '''# File: test_config.rb
    15: default[:app][:password] = nil
>>> 16: default[:app][:username] = "admin"
    17: # TODO: externalize these credentials'''
    
    print("🧪 Testing External Prompt Loader")
    print("=" * 60)
    
    # Get available versions dynamically
    available_versions = ExternalPromptLoader.get_available_versions()
    print(f"📁 Available versions: {available_versions}")
    
    for version in available_versions:
        print(f"\n📋 Testing {version.upper()} version:")
        print("-" * 40)
        
        try:
            loader = ExternalPromptLoader(version)
            
            for smell in SecuritySmell:
                print(f"\n🔍 {smell.value}:")
                prompt = loader.create_prompt(smell, test_code, PromptStyle.DEFINITION_BASED)
                print(f"Prompt length: {len(prompt)} characters")
                print(f"First 150 chars: {prompt[:150]}...")
                
        except Exception as e:
            print(f"❌ Error with {version}: {e}")


if __name__ == "__main__":
    main()