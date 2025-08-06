#!/bin/bash
# Setup script for API keys and environment variables

echo "🔧 LLM-IaC-SecEval Multi-Client Setup"
echo "======================================"

# Function to set environment variable
set_env_var() {
    local var_name=$1
    local description=$2
    local current_value=${!var_name}
    
    echo
    echo "Setting up $var_name"
    echo "Description: $description"
    
    if [ -n "$current_value" ]; then
        echo "Current value: $current_value"
        echo "Press Enter to keep current value, or type new value:"
    else
        echo "Current value: (not set)"
        echo "Enter value:"
    fi
    
    read -r new_value
    
    if [ -n "$new_value" ]; then
        export $var_name="$new_value"
        echo "export $var_name=\"$new_value\"" >> ~/.bashrc
        echo "✅ $var_name set and added to ~/.bashrc"
    elif [ -n "$current_value" ]; then
        echo "✅ Keeping current $var_name"
    else
        echo "⚠️  $var_name not set"
    fi
}

echo
echo "This script will help you set up API keys for:"
echo "• OpenAI (for GPT models)"
echo "• Ollama is local and doesn't need API keys"
echo

# OpenAI setup
set_env_var "OPENAI_API_KEY" "OpenAI API key for GPT models"

echo
echo "🧪 Testing client availability..."

# Test Ollama
echo "Checking Ollama..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama not running. Start with: ollama serve"
fi

# Test OpenAI
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OpenAI API key is set"
else
    echo "⚠️  OpenAI API key not set"
fi



echo
echo "🚀 Setup complete! You can now run:"
echo "   python scripts/validate_pipeline.py"
echo "   python scripts/run_evaluation.py --client ollama --validate-only"
echo "   python scripts/run_evaluation.py --client openai --validate-only"
echo
echo "💡 Restart your terminal or run 'source ~/.bashrc' to load new environment variables"