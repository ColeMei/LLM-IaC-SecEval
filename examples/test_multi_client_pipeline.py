#!/usr/bin/env python3
"""
Test script to demonstrate multi-client pipeline integration
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd):
    """Run a command and return success status"""
    print(f"🔄 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Success!")
            return True
        else:
            print(f"❌ Failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Timeout (30s)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_client_validation(client_type):
    """Test client validation"""
    print(f"\n{'='*50}")
    print(f"Testing {client_type.upper()} Client Validation")
    print(f"{'='*50}")
    
    cmd = [
        "python", "scripts/run_evaluation.py",
        "--client", client_type,
        "--validate-only"
    ]
    
    return run_command(cmd)

def main():
    """Test different client configurations"""
    print("🧪 Multi-Client Pipeline Integration Test")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Test all clients
    clients_to_test = ['ollama', 'openai']
    results = {}
    
    for client in clients_to_test:
        results[client] = test_client_validation(client)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for client, success in results.items():
        status = "✅ Available" if success else "❌ Not available"
        print(f"{client.upper():10} {status}")
    
    print("\n📋 Usage Examples:")
    print("   # Ollama (local)")
    print("   python scripts/run_evaluation.py --client ollama --small-batch")
    print()
    print("   # OpenAI (requires API key)")
    print("   export OPENAI_API_KEY='your-key'")
    print("   python scripts/run_evaluation.py --client openai --small-batch")
    print()
    print("   # Compare models")
    print("   python scripts/run_evaluation.py --client openai --model gpt-4 --limit 1")

if __name__ == "__main__":
    main()