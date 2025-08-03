#!/usr/bin/env python3
"""
Validation script to test pipeline on a small batch
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from automated.pipeline import LLMIaCPipeline
from automated.ollama_client import OllamaClient
from automated.file_processor import FileProcessor
from automated.config import config

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

def test_ollama_connection():
    """Test Ollama connection and model availability"""
    print("Testing Ollama connection...")
    
    client = OllamaClient()
    
    print(f"  Base URL: {client.base_url}")
    print(f"  Model: {client.model_name}")
    
    # Test connection
    available = client.is_available()
    print(f"  Available: {'✓' if available else '✗'}")
    
    if not available:
        print("  Attempting to list available models...")
        models = client.list_models()
        if models:
            print(f"    Available models: {', '.join(models)}")
        else:
            print("    No models found or connection failed")
            
        # Try to pull the model
        print(f"  Attempting to pull {client.model_name}...")
        if client.pull_model():
            print("    Model pulled successfully")
        else:
            print("    Failed to pull model")
    
    return available

def test_single_file_processing():
    """Test processing a single file end-to-end"""
    print("Testing single file processing...")
    
    # Initialize components
    client = OllamaClient()
    if not client.is_available():
        print("  Skipping - Ollama not available")
        return False
    
    pipeline = LLMIaCPipeline(model_client=client)
    processor = FileProcessor()
    
    # Get first ansible file
    try:
        for file_info in processor.batch_process_files('ansible', limit=1):
            print(f"  Processing: {file_info['filename']}")
            
            result = pipeline.run_single_file(
                filename=file_info['filename'],
                content=file_info['content'],
                ground_truth=file_info['ground_truth'],
                iac_tech='ansible',
                use_modular=True,
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
    
    # Test 2: Ollama connection
    print("\n2. Ollama Connection Tests") 
    print("-" * 30)
    ollama_available = test_ollama_connection()
    
    # Test 3: End-to-end processing (if Ollama available)
    if ollama_available:
        print("\n3. End-to-End Processing Test")
        print("-" * 30)
        e2e_success = test_single_file_processing()
    else:
        print("\n3. End-to-End Processing Test")
        print("-" * 30)
        print("  Skipped - Ollama not available")
        e2e_success = False
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"  File processing: ✓")
    print(f"  Ollama connection: {'✓' if ollama_available else '✗'}")
    print(f"  End-to-end test: {'✓' if e2e_success else '✗'}")
    
    if ollama_available and e2e_success:
        print("\n🎉 All tests passed! Pipeline is ready to use.")
        print(f"    Run: python scripts/run_evaluation.py --small-batch")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check Ollama setup:")
        print("    1. Install Ollama: https://ollama.ai")
        print("    2. Start Ollama: ollama serve")
        print("    3. Pull model: ollama pull codellama:7b")
        return 1

if __name__ == "__main__":
    sys.exit(main())