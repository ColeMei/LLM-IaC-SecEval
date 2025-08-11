#!/usr/bin/env python3
"""
Validation script to test pipeline on a small batch
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from llm_pure.pipeline import LLMIaCPipeline
from llm_pure import create_client, SUPPORTED_CLIENTS
from llm_pure.file_processor import FileProcessor
from llm_pure.config import config

def test_file_processing():
    """Test file processing components"""
    print("Testing file processing...")
    
    processor = FileProcessor()
    
    # Test dataset stats
    stats = processor.get_dataset_stats()
    print("Dataset statistics:")
    for tech, stat in stats.items():
        if 'error' in stat:
            print(f"  {tech}: Error - {stat['error']}")
        else:
            print(f"  {tech}: {stat['file_count']} files, {stat['total_annotations']} annotations")
            if stat['smell_counts']:
                top_smells = sorted(stat['smell_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"    Top smells: {', '.join(f'{s}({c})' for s, c in top_smells)}")
    
    # Test batch processing on first file
    print("\nTesting batch processing (first file only)...")
    for tech in ['ansible']:  # Just test one
        try:
            for i, file_info in enumerate(processor.batch_process_files(tech, limit=1)):
                print(f"  {tech}: {file_info['filename']} ({file_info['file_size']} bytes, {file_info['line_count']} lines)")
                print(f"    Ground truth: {len(file_info['ground_truth'])} annotations")
                break
        except Exception as e:
            print(f"  {tech}: Error - {e}")

def test_client_connection(client_type):
    """Test client connection and model availability"""
    print(f"Testing {client_type.upper()} connection...")
    
    try:
        client = create_client(client_type)
        
        print(f"  Model: {client.model_name}")
        
        # Test connection
        available = client.is_available()
        print(f"  Available: {'✓' if available else '✗'}")
        
        if not available:
            print("  Attempting to list available models...")
            models = client.list_models()
            if models:
                print(f"    Available models: {', '.join(models[:5])}...")  # Show first 5
            else:
                print("    No models found or connection failed")
                
            # For Ollama, try to pull the model
            if client_type == 'ollama' and hasattr(client, 'pull_model'):
                print(f"  Attempting to pull {client.model_name}...")
                if client.pull_model():
                    print("    Model pulled successfully")
                    available = client.is_available()  # Re-check after pull
                else:
                    print("    Failed to pull model")
        
        return available
        
    except Exception as e:
        print(f"  Error creating {client_type} client: {e}")
        if client_type == 'openai' and 'API key' in str(e):
            print("    💡 Set OPENAI_API_KEY environment variable")
        return False

def test_single_file_processing(client_type, client_available):
    """Test processing a single file end-to-end with specified client"""
    print(f"Testing single file processing with {client_type.upper()}...")
    
    if not client_available:
        print(f"  Skipping - {client_type} not available")
        return False
    
    try:
        # Initialize components
        client = create_client(client_type)
        pipeline = LLMIaCPipeline(model_client=client)
        processor = FileProcessor()
        
        # Get first ansible file
        for file_info in processor.batch_process_files('ansible', limit=1):
            print(f"  Processing: {file_info['filename']} with {client_type}")
            
            result = pipeline.run_single_file(
                filename=file_info['filename'],
                content=file_info['content'],
                ground_truth=file_info['ground_truth'],
                iac_tech='ansible',

                show_prompts=False,
                save_prompts=False,
                max_tokens=256  # Shorter for testing
            )
            
            if result['success']:
                print(f"    ✓ Success (took {result['processing_time']:.2f}s)")
                print(f"    Predictions: {len(result['predictions'])}")
                print(f"    Ground truth: {len(result['ground_truth'])}")
                if result['metrics']['overall']:
                    metrics = result['metrics']['overall']
                    print(f"    F1: {metrics['f1']:.3f}, P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}")
                return True
            else:
                print(f"    ✗ Failed: {result['error']}")
                return False
                
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    print("="*60)
    print("LLM-IaC-SecEval Pipeline Validation")
    print("="*60)
    
    # Test 1: File processing
    print("\n1. File Processing Tests")
    print("-" * 30)
    test_file_processing()
    
    # Test 2: Client connections
    print("\n2. Client Connection Tests") 
    print("-" * 30)
    client_results = {}
    for client_type in SUPPORTED_CLIENTS:
        client_results[client_type] = test_client_connection(client_type)
    
    # Test 3: End-to-end processing (test available clients)
    print("\n3. End-to-End Processing Tests")
    print("-" * 30)
    e2e_results = {}
    for client_type in SUPPORTED_CLIENTS:
        e2e_results[client_type] = test_single_file_processing(client_type, client_results[client_type])
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"  File processing: ✓")
    
    for client_type in SUPPORTED_CLIENTS:
        connection_status = '✓' if client_results[client_type] else '✗'
        e2e_status = '✓' if e2e_results[client_type] else '✗'
        print(f"  {client_type.upper()} connection: {connection_status}")
        print(f"  {client_type.upper()} end-to-end: {e2e_status}")
    
    # Determine overall success
    any_client_working = any(e2e_results.values())
    
    if any_client_working:
        working_clients = [client for client, success in e2e_results.items() if success]
        print(f"\n🎉 Pipeline validated with {', '.join(working_clients).upper()} client(s)!")
        print("    Ready to use:")
        for client in working_clients:
            print(f"      python experiments/llm_pure/run_evaluation.py --client {client} --small-batch")
        return 0
    else:
        print("\n⚠️  No clients working. Setup required:")
        
        if not client_results.get('ollama', False):
            print("\n    Ollama setup:")
            print("      1. Install Ollama: https://ollama.ai")
            print("      2. Start Ollama: ollama serve") 
            print("      3. Pull model: ollama pull codellama:7b")
        
        if not client_results.get('openai', False):
            print("\n    OpenAI setup:")
            print("      1. Get API key: https://platform.openai.com/api-keys")
            print("      2. Set environment: export OPENAI_API_KEY='your-key'")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())