"""
Configuration settings for the LLM-IaC-SecEval pipeline
"""
import os
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

@dataclass
class PipelineConfig:
    """Main configuration for the evaluation pipeline"""
    
    # Project paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"
    results_dir: Path = project_root / "results" / "llm_pure"
    prompts_dir: Path = project_root / "src" / "prompts"
    
    # Dataset configurations
    iac_technologies: List[str] = None
    oracle_datasets: Dict[str, str] = None
    
    # Model configurations
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "codellama:7b"
    
    # OpenAI configurations
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_default_model: str = "gpt-4o-mini"
    
    # Output settings
    batch_size: int = 10
    output_format: str = "csv"  # csv, json
    save_raw_responses: bool = True
    
    # Evaluation settings
    agreement_threshold: int = 1  # Include all entries (no minimum agreement filtering)
    
    # Prompt settings
    default_prompt_style: str = "definition_based"  # or "static_analysis_rules"
    
    def __post_init__(self):
        if self.iac_technologies is None:
            self.iac_technologies = ["ansible", "chef", "puppet"]
            
        if self.oracle_datasets is None:
            self.oracle_datasets = {
                "ansible": "oracle-dataset-ansible.csv",
                "chef": "oracle-dataset-chef.csv", 
                "puppet": "oracle-dataset-puppet.csv"
            }
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "raw_responses").mkdir(exist_ok=True)
        (self.results_dir / "evaluations").mkdir(exist_ok=True)
        (self.results_dir / "prompts").mkdir(exist_ok=True)

# Global config instance
config = PipelineConfig()