#!/usr/bin/env python3
"""
Main execution script for LLM-IaC-SecEval pipeline
"""
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from automated.pipeline import LLMIaCPipeline
from automated import create_client, SUPPORTED_CLIENTS
from automated.config import config

def main():
    parser = argparse.ArgumentParser(description="Run LLM-IaC-SecEval automated pipeline")
    
    # Client and model selection
    parser.add_argument("--client", choices=SUPPORTED_CLIENTS, default="ollama",
                       help=f"LLM client type (choices: {SUPPORTED_CLIENTS}, default: ollama)")
    parser.add_argument("--model", 
                       help="Model name (default varies by client)")
    
    # Client-specific options
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                       help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--openai-api-key", 
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    # Execution options
    parser.add_argument("--iac-tech", choices=["ansible", "chef", "puppet"], 
                       help="Specific IaC technology to evaluate (default: all)")
    parser.add_argument("--limit", type=int, 
                       help="Limit number of files per technology (default: all)")
    parser.add_argument("--no-modular", action="store_true",
                       help="Disable modular prompting (use full context)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Generation temperature (default: 0.1)")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum tokens to generate (default: 512)")
    
    # Modes
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate setup without running evaluation")
    parser.add_argument("--small-batch", action="store_true",
                       help="Run small batch test (5 files per tech)")
    
    # Debug options
    parser.add_argument("--show-prompts", action="store_true",
                       help="Display constructed prompts in console")
    parser.add_argument("--save-prompts", action="store_true",
                       help="Save constructed prompts to files")
    
    args = parser.parse_args()
    
    # Determine default model for each client type
    default_models = {
        'ollama': 'codellama:7b',
        'openai': 'gpt-3.5-turbo'
    }
    
    model_name = args.model or default_models[args.client]
    
    # Prepare client-specific parameters
    client_kwargs = {'model_name': model_name}
    
    if args.client == 'ollama':
        client_kwargs['base_url'] = args.ollama_url
    elif args.client == 'openai':
        if args.openai_api_key:
            client_kwargs['api_key'] = args.openai_api_key
    
    # Initialize model client using factory
    try:
        client = create_client(args.client, **client_kwargs)
    except Exception as e:
        print(f"❌ Failed to create {args.client} client: {e}")
        if args.client == 'openai' and 'API key' in str(e):
            print("   💡 Set OPENAI_API_KEY environment variable or use --openai-api-key")
        return 1
    
    # Initialize pipeline
    pipeline = LLMIaCPipeline(model_client=client)
    
    print("="*60)
    print("LLM-IaC-SecEval Automated Pipeline")
    print("="*60)
    
    # Validate setup
    print("\n1. Validating setup...")
    validation = pipeline.validate_setup()
    
    print(f"   Model available: {'✓' if validation['model_available'] else '✗'}")
    print(f"   Data directory: {'✓' if validation['data_dir_exists'] else '✗'}")
    print(f"   Results directory: {'✓' if validation['results_dir_writable'] else '✗'}")
    
    for tech, status in validation.get('datasets', {}).items():
        if 'error' in status:
            print(f"   {tech.upper()}: ✗ {status['error']}")
        else:
            print(f"   {tech.upper()}: ✓ {status['files_count']} files, {status['ground_truth_loaded']} GT")
    
    if not validation['model_available']:
        print(f"\n❌ {args.client.upper()} client/model not available!")
        if 'model_error' in validation:
            print(f"   Error: {validation['model_error']}")
        
        # Provide client-specific troubleshooting
        if args.client == 'ollama':
            print(f"   💡 Try: ollama pull {model_name}")
            print("   💡 Ensure Ollama is running: ollama serve")
        elif args.client == 'openai':
            print("   💡 Check your OpenAI API key and internet connection")
            print("   💡 Verify the model name is correct")
        return 1
    
    if args.validate_only:
        print("\n✓ Validation complete!")
        return 0
    
    # Determine execution parameters
    iac_technologies = [args.iac_tech] if args.iac_tech else config.iac_technologies
    limit = 5 if args.small_batch else args.limit
    use_modular = not args.no_modular
    
    generation_kwargs = {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens
    }
    
    print(f"\n2. Running evaluation...")
    print(f"   Technologies: {iac_technologies}")
    print(f"   Client: {args.client}")
    print(f"   Model: {model_name}")
    print(f"   Modular prompting: {use_modular}")
    print(f"   Limit per tech: {limit or 'all'}")
    print(f"   Generation params: {generation_kwargs}")
    
    try:
        # Run evaluation
        results = pipeline.run_full_evaluation(
            iac_technologies=iac_technologies,
            limit_per_tech=limit,
            use_modular=use_modular,
            show_prompts=args.show_prompts,
            save_prompts=args.save_prompts,
            **generation_kwargs
        )
        
        print(f"\n3. Evaluation complete!")
        print(f"   Experiment ID: {pipeline.experiment_id}")
        
        # Print summary
        for tech, tech_results in results['results_by_technology'].items():
            if 'overall_metrics' in tech_results:
                metrics = tech_results['overall_metrics']['overall']
                print(f"   {tech.upper()}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
            else:
                print(f"   {tech.upper()}: Error or no results")
        
        if 'cross_technology_summary' in results:
            summary = results['cross_technology_summary']
            if 'average_f1' in summary:
                print(f"   Average F1: {summary['average_f1']:.3f}")
        
        print(f"\n✓ Results saved in: {config.results_dir}")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())