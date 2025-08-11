"""
Main pipeline orchestration for LLM-IaC-SecEval
"""
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .config import config
from .prompt_builder import PromptBuilder
from .file_processor import FileProcessor
from .evaluator import Evaluator
from .model_client import ModelClient
from .ollama_client import OllamaClient

class LLMIaCPipeline:
    """Main pipeline for automated LLM evaluation on IaC security smells"""
    
    def __init__(self, model_client: ModelClient = None, prompt_style: str = None):
        self.prompt_style = prompt_style or config.default_prompt_style
        self.prompt_builder = PromptBuilder(prompt_style=self.prompt_style)
        self.file_processor = FileProcessor()
        self.evaluator = Evaluator()
        self.model_client = model_client or OllamaClient()
        
        # Results storage
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that all components are ready
        
        Returns:
            Dictionary with validation results
        """
        validation = {}
        
        # Check model availability
        try:
            validation['model_available'] = self.model_client.is_available()
        except Exception as e:
            validation['model_available'] = False
            validation['model_error'] = str(e)
        
        # Check data directories
        validation['data_dir_exists'] = config.data_dir.exists()
        
        # Check datasets
        validation['datasets'] = {}
        for iac_tech in config.iac_technologies:
            try:
                files = self.file_processor.get_iac_files(iac_tech)
                gt = self.file_processor.load_ground_truth(iac_tech)
                validation['datasets'][iac_tech] = {
                    'files_count': len(files),
                    'ground_truth_loaded': len(gt) > 0
                }
            except Exception as e:
                validation['datasets'][iac_tech] = {'error': str(e)}
        
        # Check results directory
        validation['results_dir_writable'] = config.results_dir.exists()
        
        return validation
    
    def run_single_file(self, 
                       filename: str, 
                       content: str, 
                       ground_truth: List[tuple],
                       iac_tech: str,
                       show_prompts: bool = False,
                       save_prompts: bool = False,
                       **generation_kwargs) -> Dict[str, Any]:
        """
        Process a single IaC file
        
        Args:
            filename: Name of the file
            content: File content
            ground_truth: Ground truth annotations
            iac_tech: IaC technology
            show_prompts: Whether to display constructed prompts in console
            save_prompts: Whether to save constructed prompts to files
            **generation_kwargs: Additional parameters for model generation
            
        Returns:
            Dictionary with results and metrics
        """
        start_time = time.time()
        
        # Build prompt using the configured style
        prompt = self.prompt_builder.build_prompt(filename, content)
        
        # Display prompt if requested
        if show_prompts:
            print(f"\n" + "="*80)
            print(f"CONSTRUCTED PROMPT FOR: {filename}")
            print(f"Style: {self.prompt_style}")
            print(f"Length: {len(prompt)} characters")
            print("="*80)
            print(prompt)
            print("="*80 + "\n")
        
        # Save prompt if requested
        if save_prompts:
            self._save_prompt(prompt, filename)
        
        # Generate response
        try:
            response = self.model_client.generate(prompt, **generation_kwargs)
            
            # Extract structured predictions
            predictions = self.prompt_builder.extract_response_data(response.content, filename)
            
            # Evaluate against ground truth
            metrics = self.evaluator.calculate_metrics(predictions, ground_truth)
            error_analysis = self.evaluator.analyze_errors(predictions, ground_truth)
            
            result = {
                'filename': filename,
                'iac_tech': iac_tech,
                'predictions': predictions,
                'ground_truth': ground_truth,
                'metrics': metrics,
                'error_analysis': error_analysis,
                'model_response': {
                    'content': response.content,
                    'model_name': response.model_name,
                    'response_time': response.response_time,
                    'tokens': {
                        'prompt': response.prompt_tokens,
                        'completion': response.completion_tokens,
                        'total': response.total_tokens
                    }
                },
                'processing_time': time.time() - start_time,
                'prompt_style': self.prompt_style,
                'success': True
            }
            
        except Exception as e:
            result = {
                'filename': filename,
                'iac_tech': iac_tech,
                'error': str(e),
                'success': False,
                'processing_time': time.time() - start_time
            }
        
        return result
    
    def run_batch(self, 
                  iac_tech: str, 
                  limit: int = None,
                  save_individual: bool = True,
                  show_prompts: bool = False,
                  save_prompts: bool = False,
                  **generation_kwargs) -> Dict[str, Any]:
        """
        Process a batch of files for a specific IaC technology
        
        Args:
            iac_tech: IaC technology to process
            limit: Maximum number of files to process
            save_individual: Whether to save individual results
            show_prompts: Whether to display constructed prompts in console
            save_prompts: Whether to save constructed prompts to files
            **generation_kwargs: Parameters for model generation
            
        Returns:
            Batch processing results
        """
        print(f"Starting batch processing for {iac_tech} (limit: {limit})")
        print(f"Model: {self.model_client.model_name}")
        print(f"Prompt style: {self.prompt_style}")
        
        batch_results = []
        all_predictions = []
        all_ground_truth = []
        
        # Process files
        for i, file_info in enumerate(self.file_processor.batch_process_files(iac_tech, limit)):
            print(f"Processing file {i+1}: {file_info['filename']}")
            
            result = self.run_single_file(
                filename=file_info['filename'],
                content=file_info['content'],
                ground_truth=file_info['ground_truth'],
                iac_tech=iac_tech,
                show_prompts=show_prompts,
                save_prompts=save_prompts,
                **generation_kwargs
            )
            
            batch_results.append(result)
            
            if result['success']:
                all_predictions.extend(result['predictions'])
                all_ground_truth.extend(result['ground_truth'])
                
                # Save individual result if requested
                if save_individual:
                    self._save_individual_result(result)
            else:
                print(f"Error processing {file_info['filename']}: {result['error']}")
        
        # Calculate overall batch metrics
        if all_predictions and all_ground_truth:
            overall_metrics = self.evaluator.calculate_metrics(all_predictions, all_ground_truth)
            overall_error_analysis = self.evaluator.analyze_errors(all_predictions, all_ground_truth)
        else:
            overall_metrics = {}
            overall_error_analysis = {}
        
        # Create batch summary
        batch_summary = {
            'experiment_id': self.experiment_id,
            'iac_technology': iac_tech,
            'model_name': self.model_client.model_name,
            'total_files': len(batch_results),
            'successful_files': sum(1 for r in batch_results if r['success']),
            'failed_files': sum(1 for r in batch_results if not r['success']),
            'overall_metrics': overall_metrics,
            'overall_error_analysis': overall_error_analysis,
            'individual_results': batch_results,
            'processing_metadata': {
                'prompt_style': self.prompt_style,
                'generation_params': generation_kwargs,
                'timestamp': datetime.now().isoformat(),
                'total_processing_time': sum(r.get('processing_time', 0) for r in batch_results)
            }
        }
        
        return batch_summary
    
    def run_full_evaluation(self, 
                           iac_technologies: List[str] = None,
                           limit_per_tech: int = None,
                           show_prompts: bool = False,
                           save_prompts: bool = False,
                           **generation_kwargs) -> Dict[str, Any]:
        """
        Run evaluation across all specified IaC technologies
        
        Args:
            iac_technologies: List of IaC techs to evaluate (default: all)
            limit_per_tech: Limit files per technology
            show_prompts: Whether to display constructed prompts in console
            save_prompts: Whether to save constructed prompts to files
            **generation_kwargs: Model generation parameters
            
        Returns:
            Complete evaluation results
        """
        if iac_technologies is None:
            iac_technologies = config.iac_technologies
        
        print(f"Starting full evaluation across {len(iac_technologies)} technologies")
        print(f"Model: {self.model_client.model_name}")
        
        full_results = {}
        
        for iac_tech in iac_technologies:
            print(f"\n{'='*50}")
            print(f"Processing {iac_tech.upper()}")
            print(f"{'='*50}")
            
            try:
                batch_result = self.run_batch(
                    iac_tech=iac_tech,
                    limit=limit_per_tech,
                    show_prompts=show_prompts,
                    save_prompts=save_prompts,
                    **generation_kwargs
                )
                full_results[iac_tech] = batch_result
                
                # Save batch results
                self._save_batch_results(batch_result, iac_tech)
                
            except Exception as e:
                print(f"Error processing {iac_tech}: {e}")
                full_results[iac_tech] = {'error': str(e)}
        
        # Create comprehensive evaluation report
        evaluation_report = self._create_full_evaluation_report(full_results)
        
        # Save complete evaluation
        self._save_full_evaluation(evaluation_report)
        
        return evaluation_report
    
    def _save_prompt(self, prompt: str, filename: str):
        """Save constructed prompt to file"""
        timestamp = datetime.now().strftime("%H%M%S")
        prompt_filename = f"prompt_{self.prompt_style}_{filename}_{timestamp}.txt"
        output_path = config.results_dir / "prompts" / prompt_filename
        
        # Ensure prompts directory exists
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# CONSTRUCTED PROMPT\n")
            f.write(f"# File: {filename}\n")
            f.write(f"# Style: {self.prompt_style}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"# Length: {len(prompt)} characters\n")
            f.write(f"# {'='*60}\n\n")
            f.write(prompt)
    
    def _save_individual_result(self, result: Dict[str, Any]):
        """Save individual file result"""
        if config.save_raw_responses:
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"{result['filename']}_{timestamp}.json"
            output_path = config.results_dir / "raw_responses" / filename
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
    
    def _save_batch_results(self, batch_result: Dict[str, Any], iac_tech: str):
        """Save batch processing results"""
        filename = f"batch_{iac_tech}_{self.experiment_id}.json"
        output_path = config.results_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(batch_result, f, indent=2, default=str)
        
        print(f"Batch results saved to: {output_path}")
    
    def _save_full_evaluation(self, evaluation_report: Dict[str, Any]):
        """Save complete evaluation report"""
        filename = f"full_evaluation_{self.experiment_id}.json"
        output_path = config.results_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        print(f"Full evaluation saved to: {output_path}")
    
    def _create_full_evaluation_report(self, full_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive evaluation report"""
        return {
            'experiment_info': {
                'experiment_id': self.experiment_id,
                'model_name': self.model_client.model_name,
                'timestamp': datetime.now().isoformat(),
                'technologies_evaluated': list(full_results.keys())
            },
            'results_by_technology': full_results,
            'cross_technology_summary': self._calculate_cross_tech_summary(full_results)
        }
    
    def _calculate_cross_tech_summary(self, full_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics across all technologies"""
        all_f1_scores = []
        all_precision_scores = []
        all_recall_scores = []
        
        for iac_tech, results in full_results.items():
            if 'overall_metrics' in results and 'overall' in results['overall_metrics']:
                metrics = results['overall_metrics']['overall']
                all_f1_scores.append(metrics['f1'])
                all_precision_scores.append(metrics['precision'])
                all_recall_scores.append(metrics['recall'])
        
        if all_f1_scores:
            return {
                'average_f1': sum(all_f1_scores) / len(all_f1_scores),
                'average_precision': sum(all_precision_scores) / len(all_precision_scores),
                'average_recall': sum(all_recall_scores) / len(all_recall_scores),
                'best_technology': max(full_results.items(), 
                                     key=lambda x: x[1].get('overall_metrics', {}).get('overall', {}).get('f1', 0))[0],
                'worst_technology': min(full_results.items(), 
                                      key=lambda x: x[1].get('overall_metrics', {}).get('overall', {}).get('f1', 1))[0]
            }
        else:
            return {'error': 'No valid metrics calculated'}