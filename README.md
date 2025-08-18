# LLM-IaC-SecEval: Evaluating Large Language Models for Infrastructure-as-Code Security Analysis

> **Draft Research Project Overview**

## Project Overview

This research project investigates the effectiveness of Large Language Models (LLMs) in detecting security vulnerabilities within Infrastructure-as-Code (IaC) scripts. We explore how LLMs can be leveraged to identify security anti-patterns (security smells) in configuration management tools like Ansible, Chef, and Puppet.

## Research Goals

The primary objective is to evaluate and compare different approaches for automated security smell detection in IaC scripts, focusing on:

- **Accuracy**: How well can LLMs identify genuine security issues?
- **Precision vs Recall Trade-offs**: Understanding the balance between catching all issues and minimizing false positives
- **Practical Applicability**: Developing approaches suitable for real-world DevOps workflows

## Security Smells Under Investigation

Our research focuses on detecting **9 categories of security smells** commonly found in IaC scripts:

1. **Admin by default** - Granting unnecessary administrative privileges
2. **Empty password** - Using zero-length passwords
3. **Hard-coded secret** - Embedding sensitive information directly in scripts
4. **Missing default in case statement** - Incomplete conditional logic handling
5. **No integrity check** - Downloading content without checksum verification
6. **Suspicious comment** - Comments indicating known defects (TODO, FIXME, HACK)
7. **Unrestricted IP address** - Using 0.0.0.0 exposing services broadly
8. **Use of HTTP without SSL/TLS** - Insecure communication protocols
9. **Use of weak cryptography algorithms** - Employing deprecated encryption methods

## Research Approaches

### 1. Pure LLM Evaluation

**Objective**: Assess LLMs' standalone capability to detect all 9 security smell categories.

**Methodology**:

- Direct prompting of state-of-the-art LLMs (GPT-4, Claude, Ollama models)
- Multiple configurable prompt templates for optimization
- Systematic evaluation across comprehensive IaC script datasets
- Analysis of detection accuracy, precision, and recall for each security smell type
- Investigation of prompt engineering techniques for optimal performance

**Focus**: Understanding the inherent strengths and limitations of LLMs for security analysis without external tool assistance.

### 2. Hybrid Evaluation (LLM + Static Analysis)

**Objective**: Combine static analysis tools with LLMs to achieve superior detection performance.

**Methodology**:

- **Stage 1**: Use GLITCH (static analysis tool) for initial detection with high recall
- **Stage 2**: Apply LLMs as intelligent post-filters to reduce false positives
- Focus on **4 specific security smells** where semantic understanding provides value:
  - Hard-coded secrets (distinguish real secrets from placeholders)
  - Suspicious comments (identify security-relevant vs. general comments)
  - Weak cryptography (detect actual usage vs. documentation mentions)
  - Use of HTTP without SSL/TLS (identify insecure communication patterns)
- Configurable prompt templates and context extraction
- Support for multiple LLM providers (OpenAI, Anthropic, Ollama, OpenAI-compatible APIs)

**Rationale**: Static tools excel at comprehensive pattern matching (high recall) but generate many false alarms (low precision). LLMs can provide contextual understanding to filter out false positives while preserving true security issues.

## Current Project Structure

```
LLM-IaC-SecEval/
├── data/                          # Original datasets
├── docs/                          # Documentation
├── experiments/                   # Execution interface
│   ├── llm_pure/                 # Pure LLM method execution (scripts)
│   ├── llm_postfilter/           # Hybrid method execution (notebooks + data)
│   ├── data-analysis/            # Analysis notebooks
│   └── zero-shot/                # Zero-shot experiments
├── results/                       # All results from both methodologies
│   ├── llm_pure/                 # Results from pure LLM method
│   └── llm_postfilter/           # Results from hybrid method
└── src/                           # Code logic only
    ├── llm_pure/                 # Pure LLM implementation
    ├── llm_postfilter/           # Hybrid implementation
    └── prompts/                  # Prompt templates
```

## Research Status

This is an **ongoing research project** in its initial stages. We are currently developing and refining both evaluation approaches to establish baseline performance metrics and identify optimal methodologies for LLM-assisted IaC security analysis.

## Future Work

- Comprehensive evaluation across multiple IaC tools and larger datasets
- Investigation of different LLM architectures and prompt strategies
- Development of production-ready tools for DevOps integration
- Comparative analysis with other automated security scanning solutions

---

_This research contributes to the broader field of AI-assisted cybersecurity and secure software development practices._
