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
    HTTP_WITHOUT_SSL_TLS = "Use of HTTP without SSL/TLS"


class PromptStyle(Enum):
    """Enumeration of available prompt styles."""
    DEFINITION_BASED = "definition_based"
    STATIC_ANALYSIS_RULES = "static_analysis_rules"


"""Prompt versions are represented by free-form suffixes in filenames.
We intentionally avoid an enum to make adding new versions simpler.
"""


class ExternalPromptLoader:
    """Loads prompts and definitions from external files using template names."""
    
    def __init__(self, prompt_template: str = "definition_based_conservative"):
        """Initialize with specified prompt template."""
        self.prompt_template = prompt_template
        self.project_root = Path(__file__).parent.parent.parent
        self.prompts_dir = self.project_root / "src" / "prompts" / "llm_postfilter"
        
        # Parse template to extract style and version (version is optional and free-form)
        if prompt_template.startswith("definition_based"):
            self.style = PromptStyle.DEFINITION_BASED
            if prompt_template == "definition_based":
                self.version = None
            elif prompt_template.startswith("definition_based_"):
                self.version = prompt_template.replace("definition_based_", "", 1)
            else:
                raise ValueError(f"Invalid prompt template: {prompt_template}")
        elif prompt_template.startswith("static_analysis_rules"):
            self.style = PromptStyle.STATIC_ANALYSIS_RULES
            if prompt_template == "static_analysis_rules":
                self.version = None
            elif prompt_template.startswith("static_analysis_rules_"):
                self.version = prompt_template.replace("static_analysis_rules_", "", 1)
            else:
                raise ValueError(f"Invalid prompt template: {prompt_template}")
        else:
            raise ValueError(f"Invalid prompt template: {prompt_template}")
        
        # Load definitions once during initialization
        self._definitions = self._load_smell_definitions()
    
    def _get_template_filename(self) -> Path:
        """Resolve the prompt template filename based on style and optional version.
        Uses names without the "_prompt" suffix, e.g., definition_based.txt or definition_based_<version>.txt
        """
        if self.style == PromptStyle.DEFINITION_BASED:
            if self.version is None:
                return self.prompts_dir / "definition_based.txt"
            return self.prompts_dir / f"definition_based_{self.version}.txt"
        else:
            if self.version is None:
                return self.prompts_dir / "static_analysis_rules.txt"
            return self.prompts_dir / f"static_analysis_rules_{self.version}.txt"
    
    def _get_yaml_filename(self, base_name: str) -> Path:
        """Get YAML filename for smell definitions. Uses versionless default when self.version is None."""
        if self.version is None:
            return self.prompts_dir / f"{base_name}.yaml"
        return self.prompts_dir / f"{base_name}_{self.version}.yaml"
    
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
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from file."""
        template_file = self._get_template_filename()
        
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")
        
        return template_file.read_text(encoding='utf-8')
    
    def get_definition(self, smell: SecuritySmell) -> str:
        """Get definition for a specific security smell."""
        return self._definitions[smell]
    
    def create_prompt(self, smell: SecuritySmell, code_snippet: str) -> str:
        """Create a complete prompt for a specific security smell and code snippet."""
        template = self._load_prompt_template()
        
        if self.style == PromptStyle.DEFINITION_BASED:
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
    
    def get_base_prompt(self) -> str:
        """Get the base prompt template."""
        return self._load_prompt_template()
    
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
    def get_available_templates(cls) -> list:
        """Get list of available prompt templates by scanning filenames.
        Includes versionless defaults and any versioned variants (free-form suffixes)."""
        prompts_dir = Path(__file__).parent.parent.parent / "src" / "prompts" / "llm_postfilter"
        templates = set()

        # Versionless defaults
        if (prompts_dir / "definition_based.txt").exists():
            templates.add("definition_based")
        if (prompts_dir / "static_analysis_rules.txt").exists():
            templates.add("static_analysis_rules")

        # Versioned variants (free-form)
        for file in prompts_dir.glob("definition_based_*.txt"):
            # Skip the base file handled above
            if file.name == "definition_based.txt":
                continue
            version = file.stem.replace("definition_based_", "", 1)
            if version:
                templates.add(f"definition_based_{version}")
        for file in prompts_dir.glob("static_analysis_rules_*.txt"):
            if file.name == "static_analysis_rules.txt":
                continue
            version = file.stem.replace("static_analysis_rules_", "", 1)
            if version:
                templates.add(f"static_analysis_rules_{version}")

        return sorted(list(templates))
    
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