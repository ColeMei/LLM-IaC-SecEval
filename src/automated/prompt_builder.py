"""
Modular prompt builder for IaC security smell detection with automated context separation
"""
from pathlib import Path
from typing import Dict, Optional
from .config import config

class PromptBuilder:
    """Builds prompts with modular background context separation"""
    
    def __init__(self):
        self.background_context = self._load_security_smells_context()
        self.instruction_template = self._load_instruction_template()
    
    def _load_security_smells_context(self) -> str:
        """Load security smells definitions as background context"""
        smells_file = config.data_dir / "smells-description.txt"
        if smells_file.exists():
            return smells_file.read_text(encoding='utf-8')
        else:
            raise FileNotFoundError(f"Security smells definitions file not found: {smells_file}")
    
    def _load_instruction_template(self) -> str:
        """Load the core instruction template (instructions only, no security smell definitions)"""
        # For modular approach, use instructions-only template to avoid duplication
        template_file = config.prompts_dir / "Template_instructions_only.txt"
        if template_file.exists():
            return template_file.read_text(encoding='utf-8').strip()
        else:
            # Fallback to extracting from detailed template
            detailed_template_file = config.prompts_dir / "Template_detailed.txt"
            if detailed_template_file.exists():
                content = detailed_template_file.read_text(encoding='utf-8')
                # Extract only the instruction part (before security smell definitions)
                parts = content.split("# Definitions of Security Smells")
                return parts[0].strip()
            else:
                raise FileNotFoundError(f"No instruction template file found: tried {template_file} and {detailed_template_file}")

    def build_prompt(self, filename: str, file_content: str, include_context: bool = True) -> str:
        """
        Build a complete prompt for LLM evaluation
        
        Args:
            filename: Name of the IaC file
            file_content: Content of the IaC file
            include_context: Whether to use modular approach (True) or full context approach (False)
            
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        if include_context:
            # Modular approach: Separate background context + instructions
            prompt_parts.append("# Background Context")
            prompt_parts.append(self.background_context)
            prompt_parts.append("")
            prompt_parts.append(self.instruction_template)
        else:
            # Full context approach: Use complete template with embedded definitions
            full_template_file = config.prompts_dir / "Template_detailed.txt"
            if full_template_file.exists():
                full_template = full_template_file.read_text(encoding='utf-8').strip()
                prompt_parts.append(full_template)
            else:
                # Fallback to modular if detailed template not found
                prompt_parts.append("# Background Context")
                prompt_parts.append(self.background_context)
                prompt_parts.append("")
                prompt_parts.append(self.instruction_template)
        
        prompt_parts.append("")
        
        # Add the specific file to analyze
        prompt_parts.append("# Script for Analysis")
        prompt_parts.append("")
        prompt_parts.append(f"File name: {filename}")
        prompt_parts.append(f"File content:")
        prompt_parts.append("```")
        prompt_parts.append(file_content)
        prompt_parts.append("```")
        
        return "\n".join(prompt_parts)
    
    def extract_response_data(self, response: str, filename: str) -> list:
        """
        Extract structured data from LLM response
        
        Args:
            response: Raw LLM response text
            filename: Original filename for validation
            
        Returns:
            List of tuples: (filename, line_number, category)
        """
        findings = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or 'NAME_OF_FILE' in line:
                continue
                
            # Try to parse CSV format
            try:
                parts = [part.strip().strip('"').strip("'").strip("{}") for part in line.split(',')]
                if len(parts) >= 3:
                    file_part, line_num, category = parts[0], parts[1], parts[2]
                    
                    # Validate line number
                    try:
                        line_number = int(line_num)
                    except ValueError:
                        continue
                        
                    findings.append((filename, line_number, category.strip()))
                        
            except Exception:
                # Skip malformed lines
                continue
                
        # If no valid findings found, assume no smells
        if not findings:
            findings.append((filename, 0, "none"))
            
        return findings