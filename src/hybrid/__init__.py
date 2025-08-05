"""
Hybrid LLM + Static Analysis Approach

This module implements a two-stage detection pipeline that combines:
1. Static analysis tools (GLITCH) for high recall detection
2. LLM post-filtering for improved precision

The hybrid approach leverages the strengths of both:
- Static tools: Conservative, high recall, pattern-based detection
- LLMs: Contextual understanding, semantic reasoning, false positive reduction
"""

from .data_extractor import GLITCHDetectionExtractor

__all__ = ['GLITCHDetectionExtractor']