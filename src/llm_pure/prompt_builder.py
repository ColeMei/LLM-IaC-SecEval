"""
Simplified prompt builder for IaC security smell detection with two distinct styles
"""
from pathlib import Path
from typing import Dict, Optional, List
from .config import config

class PromptBuilder:
    """Builds prompts with two distinct styles: definition_based and static_analysis_rules"""
    
    def __init__(self, prompt_style: str = 'definition_based'):
        from . import SUPPORTED_PROMPT_STYLES
        if prompt_style not in SUPPORTED_PROMPT_STYLES:
            raise ValueError(f"Unsupported prompt style: {prompt_style}. Supported styles: {SUPPORTED_PROMPT_STYLES}")
        
        self.prompt_style = prompt_style
        self.template = self._load_template()
    
    def _load_template(self) -> str:
        """Load the template for the selected prompt style"""
        template_file = config.prompts_dir / f"Template_{self.prompt_style}.txt"
        
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")
        
        return template_file.read_text(encoding='utf-8')
    
    def _detect_iac_language(self, filename: str) -> str:
        """Detect IaC language based on file extension"""
        file_extension = Path(filename).suffix.lower()
        
        language_map = {
            '.tf': 'Terraform',
            '.hcl': 'Terraform',
            '.yml': 'Ansible/YAML',
            '.yaml': 'Ansible/YAML',
            '.pp': 'Puppet',
            '.py': 'Python',
            '.ps1': 'PowerShell',
            '.sh': 'Shell/Bash',
            '.json': 'JSON',
            '.dockerfile': 'Docker',
            '.docker': 'Docker'
        }
        
        return language_map.get(file_extension, 'Unknown')
    
    def _add_line_numbers(self, content: str) -> str:
        """Add line numbers to code content for static analysis rules style"""
        lines = content.split('\n')
        numbered_lines = []
        
        for i, line in enumerate(lines, 1):
            numbered_lines.append(f"Line {i}: {line}")
        
        return '\n'.join(numbered_lines)
    
    def build_prompt(self, filename: str, file_content: str) -> str:
        """
        Build a complete prompt for LLM evaluation
        
        Args:
            filename: Name of the IaC file
            file_content: Content of the IaC file
            
        Returns:
            Complete prompt string
        """
        iac_language = self._detect_iac_language(filename)
        
        if self.prompt_style == 'definition_based':
            return self._build_definition_based_prompt(filename, file_content, iac_language)
        elif self.prompt_style == 'static_analysis_rules':
            return self._build_static_analysis_prompt(filename, file_content, iac_language)
        else:
            raise ValueError(f"Unknown prompt style: {self.prompt_style}")
    
    def _build_definition_based_prompt(self, filename: str, file_content: str, iac_language: str) -> str:
        """Build definition-based prompt"""
        # Add line numbers to the content
        numbered_content = self._add_line_numbers(file_content)
        
        # Replace the placeholder in the template with actual content
        prompt = self.template.replace(
            "Line 1: class govuk_beat::repo (\nLine 2:   $apt_mirror_hostname,\nLine 3: ) {\nLine 4:   apt::source { 'elastic-beats':\nLine 5:     location     => \"http://${apt_mirror_hostname}/elastic-beats\",\nLine 6:     release      => 'stable',\nLine 7:     architecture => $::architecture,\nLine 8:   }\nLine 9: }",
            numbered_content
        )
        
        # Add file name information
        file_info = f"\n\n**Analyzing file: {filename} ({iac_language})**\n"
        
        # Insert file info before the RAW CODE INPUT section
        prompt = prompt.replace("### RAW CODE INPUT", f"### RAW CODE INPUT{file_info}")
        
        return prompt
    
    def _build_static_analysis_prompt(self, filename: str, file_content: str, iac_language: str) -> str:
        """Build static analysis rules-based prompt"""
        # Add line numbers to the content
        numbered_content = self._add_line_numbers(file_content)
        
        # Replace the placeholder in the template with actual content
        prompt = self.template.replace(
            "Line 1: class govuk_beat::repo (\nLine 2:   $apt_mirror_hostname,\nLine 3: ) {\nLine 4:   apt::source { 'elastic-beats':\nLine 5:     location     => \"http://${apt_mirror_hostname}/elastic-beats\",\nLine 6:     release      => 'stable',\nLine 7:     architecture => $::architecture,\nLine 8:   }\nLine 9: }",
            numbered_content
        )
        
        # Add file name information
        file_info = f"\n\n**Analyzing file: {filename} ({iac_language})**\n"
        
        # Insert file info before the RAW CODE INPUT section
        prompt = prompt.replace("### RAW CODE INPUT", f"### RAW CODE INPUT{file_info}")
        
        return prompt
    
    def extract_response_data(self, response: str, filename: str) -> List[tuple]:
        """
        Extract structured data from LLM response for both prompt styles
        
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
                    
                    # Normalize category names for static analysis rules style
                    category = self._normalize_category_name(category.strip())
                    findings.append((filename, line_number, category))
                        
            except Exception:
                # Skip malformed lines
                continue
        
        # If no valid findings found, assume no smells
        if not findings:
            findings.append((filename, 0, "none"))
            
        return findings
    
    def _normalize_category_name(self, category: str) -> str:
        """Normalize category names to match expected format"""
        # Mapping for static analysis rules style to standard names
        category_mappings = {
            "Invalid IP address binding": "Unrestricted IP Address",
            "Use of HTTP without TLS": "Use of HTTP without SSL/TLS",
            "Use of weak crypto algorithm": "Use of weak cryptography algorithms",
            "Missing default case statement": "Missing Default in Case Statement"
        }
        
        return category_mappings.get(category, category)
    
    def get_style_info(self) -> Dict[str, str]:
        """Get information about the current prompt style"""
        return {
            'style': self.prompt_style,
            'description': {
                'definition_based': 'Uses human-readable definitions of security smells',
                'static_analysis_rules': 'Uses formal logical rules inspired by GLITCH paper'
            }.get(self.prompt_style, 'Unknown style')
        }