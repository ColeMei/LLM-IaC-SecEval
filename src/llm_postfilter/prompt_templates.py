"""
LLM Prompt Templates for Security Smell Detection

This module contains formal definitions and prompt templates for each security smell category.
Templates are designed for GPT-4o mini to make accurate YES/NO decisions on GLITCH detections.
"""

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


class SecuritySmellPrompts:
    """Container for security smell definitions and prompt templates."""
    
    # Formal definitions for each security smell
    DEFINITIONS = {
        SecuritySmell.HARD_CODED_SECRET: """
A hard-coded secret is a security vulnerability where sensitive information such as passwords, API keys, tokens, certificates, or other credentials are directly embedded in the source code as literal strings or variables, rather than being securely stored and retrieved from external configuration systems, environment variables, or secret management services.

Key characteristics:
- Passwords, API keys, tokens written directly in code
- Database connection strings with embedded credentials
- Cryptographic keys or certificates as string literals
- Authentication tokens hardcoded as variables
- Any sensitive value that should be externalized but is embedded in code

NOT hard-coded secrets:
- Placeholder values (e.g., "your-api-key-here", "REPLACE_ME")
- Example/demo credentials in comments or documentation
- Variable names that suggest secrets but contain non-sensitive values
- Configuration parameter names without actual secret values
""",

        SecuritySmell.SUSPICIOUS_COMMENT: """
A suspicious comment is a code comment that indicates potential security issues, incomplete security implementations, or areas requiring security attention. These comments often signal unfinished work, security bypasses, or acknowledged vulnerabilities that may pose risks.

Key characteristics:
- TODO/FIXME comments related to security, authentication, authorization
- Comments indicating disabled security features or bypassed checks
- Warnings about insecure code or temporary security workarounds
- Comments suggesting hardcoded values should be externalized
- References to known security issues or vulnerabilities that need addressing

NOT suspicious comments:
- General TODO/FIXME for non-security functionality
- Documentation comments explaining legitimate security implementations
- Standard code documentation without security implications
- Performance or feature-related todos
- Comments that acknowledge and explain secure design decisions
""",

        SecuritySmell.WEAK_CRYPTOGRAPHY: """
Use of weak cryptography algorithms refers to the implementation or configuration of cryptographic functions that are known to be vulnerable, deprecated, or insufficient for current security standards. This includes both algorithmically weak ciphers and poor cryptographic practices.

Key characteristics:
- Use of broken hash functions (MD5, SHA1) for security purposes
- Weak encryption algorithms (DES, 3DES, RC4)
- Insufficient key sizes for current standards
- Use of deprecated SSL/TLS versions (SSLv2, SSLv3, TLS 1.0)
- Weak random number generation for cryptographic purposes
- Improper cryptographic configurations

NOT weak cryptography:
- Use of weak algorithms for non-security purposes (checksums, legacy compatibility)
- References to weak algorithms in documentation or comments
- Configuration of strong, modern cryptographic algorithms
- Variable names mentioning algorithms without actual implementation
- Proper use of strong cryptographic libraries and functions
"""
    }
    
    @classmethod
    def get_definition_based_prompt(cls) -> str:
        """Get the definition-based prompt template."""
        return """You are an expert in Infrastructure-as-Code (IaC) security analysis specializing in Chef cookbooks, Puppet manifests, and Ansible playbooks.

A static analysis tool has flagged a potential security issue in an IaC script. Your task is to determine whether this detection represents a true instance of the specified security smell.

Instructions:
1. Carefully examine the provided code snippet and surrounding context
2. Consider the code's intent, context, and actual implementation
3. Apply the formal definition provided below
4. Focus on actual security implications, not just keyword presence
5. Answer with ONLY "YES" or "NO" - no explanations needed

{smell_definition}

Code snippet to analyze:
{code_snippet}

Based on the definition above and the code context, is this a true instance of "{smell_name}"?

Answer (YES or NO only):"""

    @classmethod
    def get_static_analysis_rules_prompt(cls) -> str:
        """Get the static analysis rules-based prompt template."""
        return """You are a static analyzer for Infrastructure-as-Code (IaC) scripts.

Your task is to analyze the **raw code** of an IaC script (e.g. Chef, Puppet) and detect any **security smells** according to the rules and keyword-based functions below.

You must parse the script internally, identify patterns, and output a list of security smells with corresponding line numbers and code excerpts.

---

### SECURITY SMELL DETECTION RULES

Smell Name: Hard-coded secret  
Rule: (isAttribute(x) ∨ isVariable(x)) ∧ (isPassword(x.name) ∨ isSecret(x.name) ∨ isUser(x.name)) ∧ ¬x.has_variable

Smell Name: Suspicious comment  
Rule: isComment(x) ∧ hasWrongWords(x.content)

Smell Name: Use of weak crypto algorithm  
Rule: (isAttribute(x) ∨ isVariable(x)) ∧ isWeakCrypt(x.value) ∧ ¬hasWeakCryptWhitelist(x.name) ∧ ¬hasWeakCryptWhitelist(x.value)

---

### STRING PATTERN MATCHING FUNCTIONS

Use the following keyword heuristics to apply the detection rules:

- isUser(): "user", "uname", "username", "login", "userid", "loginid"  
- isPassword(): "pass", "pwd", "password", "passwd", "passno", "pass-no"  
- isSecret(): "auth_token", "authentication_token", "secret", "ssh_key"  
- hasWrongWords(): "bug", "debug", "todo", "hack", "solve", "fixme"  
- isWeakCrypt(): "md5", "sha1", "arcfour"  
- hasWeakCryptWhitelist(): "checksum"

---

### INSTRUCTIONS

1. Analyze the following **raw IaC code** line-by-line.
2. For the specific smell "{smell_name}", determine if it matches the detection rule.
3. Focus on actual security implications, not just keyword presence.
4. Answer with ONLY "YES" or "NO" - no explanations needed.

---

### RAW CODE INPUT

{code_snippet}

---

Based on the static analysis rules above, does this code contain a true instance of "{smell_name}"?

Answer (YES or NO only):"""

    @classmethod
    def get_base_prompt(cls, style: PromptStyle = PromptStyle.DEFINITION_BASED) -> str:
        """Get the base prompt template for the specified style."""
        if style == PromptStyle.STATIC_ANALYSIS_RULES:
            return cls.get_static_analysis_rules_prompt()
        else:
            return cls.get_definition_based_prompt()

    @classmethod
    def create_prompt(
        cls, 
        smell: SecuritySmell, 
        code_snippet: str, 
        style: PromptStyle = PromptStyle.DEFINITION_BASED
    ) -> str:
        """Create a complete prompt for a specific security smell and code snippet."""
        if style == PromptStyle.STATIC_ANALYSIS_RULES:
            base_prompt = cls.get_static_analysis_rules_prompt()
            return base_prompt.format(
                smell_name=smell.value,
                code_snippet=code_snippet
            )
        else:
            # Definition-based style (default/backward compatible)
            base_prompt = cls.get_definition_based_prompt()
            definition = cls.DEFINITIONS[smell]
            return base_prompt.format(
                smell_definition=definition,
                smell_name=smell.value,
                code_snippet=code_snippet
            )
    
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
    def get_validation_prompt(cls, smell: SecuritySmell, style: PromptStyle = PromptStyle.DEFINITION_BASED) -> str:
        """Get a validation prompt to test LLM understanding of the smell definition."""
        if style == PromptStyle.STATIC_ANALYSIS_RULES:
            return f"""Please confirm your understanding of the static analysis rules for "{smell.value}":

Based on the rule-based detection approach, do you understand how to identify {smell.value} using logical conditions and keyword matching functions?

Please respond with "YES" if you understand, or ask for clarification if needed."""
        else:
            definition = cls.DEFINITIONS[smell]
            return f"""Please confirm your understanding of "{smell.value}":

{definition}

Do you understand this definition? Please respond with "YES" if you understand, or ask for clarification if needed."""


class PromptValidator:
    """Validates and tests prompt effectiveness."""
    
    @staticmethod
    def create_test_cases() -> Dict[SecuritySmell, Dict[str, str]]:
        """Create test cases for each security smell for prompt validation."""
        return {
            SecuritySmell.HARD_CODED_SECRET: {
                "positive_example": """
# File: database_config.rb
    15: database_password = "MySecretPassword123"
>>> 16: connection_string = "mongodb://admin:#{database_password}@localhost:27017/mydb"
    17: Chef::Log.info("Connecting to database")
""",
                "negative_example": """
# File: config_template.rb  
    12: # TODO: Replace with actual API key from environment
>>> 13: api_key = "YOUR-API-KEY-HERE"
    14: # This is just a placeholder value
"""
            },
            
            SecuritySmell.SUSPICIOUS_COMMENT: {
                "positive_example": """
# File: auth_config.rb
    8: # TODO: Fix authentication bypass - currently allows any user
>>> 9: # WARNING: Security issue - remove this temporary hack
    10: if ENV['SKIP_AUTH'] == 'true'
""",
                "negative_example": """
# File: ui_component.rb
    25: # TODO: Add better error handling for network timeouts
>>> 26: # FIXME: Improve user interface responsiveness
    27: def handle_network_error
"""
            },
            
            SecuritySmell.WEAK_CRYPTOGRAPHY: {
                "positive_example": """
# File: encryption.rb
    20: require 'digest'
>>> 21: password_hash = Digest::MD5.hexdigest(user_password)
    22: Chef::Log.debug("Password hashed for storage")
""",
                "negative_example": """
# File: documentation.rb
    5: # This cookbook supports multiple hash algorithms:
>>> 6: # - MD5 (legacy compatibility only, not recommended)  
    7: # - SHA256 (recommended for new deployments)
"""
            }
        }


def main():
    """Test prompt generation with both styles."""
    # Test prompt creation for each smell with both styles
    test_cases = PromptValidator.create_test_cases()
    
    for smell, examples in test_cases.items():
        print(f"\n{'='*60}")
        print(f"Testing {smell.value}")
        print(f"{'='*60}")
        
        for style in PromptStyle:
            print(f"\n--- {style.value.upper().replace('_', ' ')} STYLE ---")
            
            # Test positive example
            prompt = SecuritySmellPrompts.create_prompt(smell, examples["positive_example"], style)
            print(f"\nPositive Example Prompt (should be YES):")
            print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
            
            # Test negative example  
            prompt = SecuritySmellPrompts.create_prompt(smell, examples["negative_example"], style)
            print(f"\nNegative Example Prompt (should be NO):")
            print(prompt[:300] + "..." if len(prompt) > 300 else prompt)


if __name__ == "__main__":
    main()