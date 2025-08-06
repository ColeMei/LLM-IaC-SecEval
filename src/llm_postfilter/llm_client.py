"""
LLM Client for Security Smell Post-Filtering

This module provides an interface to GPT-4o mini for evaluating GLITCH detections.
Handles API calls, response parsing, and decision extraction for the hybrid pipeline.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

logger = logging.getLogger(__name__)


class LLMDecision(Enum):
    """LLM decision outcomes."""
    YES = "YES"
    NO = "NO"
    UNCLEAR = "UNCLEAR"
    ERROR = "ERROR"


@dataclass
class LLMResponse:
    """Container for LLM response data."""
    decision: LLMDecision
    raw_response: str
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None


class GPT4OMiniClient:
    """Client for GPT-4o mini API calls with security smell evaluation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize the GPT-4o mini client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        
        # Rate limiting and retry settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.requests_per_minute = 60
        self.last_request_time = 0
        
        logger.info(f"Initialized GPT-4o mini client with model: {model}")
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.requests_per_minute
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _parse_decision(self, response_text: str) -> LLMDecision:
        """Parse LLM response to extract YES/NO decision."""
        response_upper = response_text.strip().upper()
        
        # Direct matches
        if response_upper == "YES":
            return LLMDecision.YES
        elif response_upper == "NO":
            return LLMDecision.NO
        
        # Look for YES/NO in the response
        if "YES" in response_upper and "NO" not in response_upper:
            return LLMDecision.YES
        elif "NO" in response_upper and "YES" not in response_upper:
            return LLMDecision.NO
        
        # Handle ambiguous responses
        logger.warning(f"Unclear LLM response: {response_text}")
        return LLMDecision.UNCLEAR
    
    def evaluate_detection(self, prompt: str, max_tokens: int = 50) -> LLMResponse:
        """Evaluate a single detection using the LLM."""
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                self._enforce_rate_limit()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert security analyst. Answer only YES or NO based on the provided criteria."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,  # Deterministic responses
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                raw_response = response.choices[0].message.content.strip()
                decision = self._parse_decision(raw_response)
                processing_time = time.time() - start_time
                
                # Extract token usage if available
                tokens_used = None
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = response.usage.total_tokens
                
                return LLMResponse(
                    decision=decision,
                    raw_response=raw_response,
                    processing_time=processing_time,
                    tokens_used=tokens_used
                )
                
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    # Final attempt failed
                    processing_time = time.time() - start_time
                    return LLMResponse(
                        decision=LLMDecision.ERROR,
                        raw_response="",
                        processing_time=processing_time,
                        error_message=str(e)
                    )
        
        # Should not reach here, but just in case
        return LLMResponse(
            decision=LLMDecision.ERROR,
            raw_response="",
            error_message="Max retries exceeded"
        )
    
    def batch_evaluate(self, prompts: List[str], progress_callback=None) -> List[LLMResponse]:
        """Evaluate multiple detections with progress tracking."""
        responses = []
        total_prompts = len(prompts)
        
        logger.info(f"Starting batch evaluation of {total_prompts} prompts")
        
        for i, prompt in enumerate(prompts):
            response = self.evaluate_detection(prompt)
            responses.append(response)
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total_prompts, response)
            
            # Log progress periodically
            if (i + 1) % 10 == 0 or (i + 1) == total_prompts:
                yes_count = sum(1 for r in responses if r.decision == LLMDecision.YES)
                no_count = sum(1 for r in responses if r.decision == LLMDecision.NO)
                error_count = sum(1 for r in responses if r.decision == LLMDecision.ERROR)
                logger.info(f"Progress: {i + 1}/{total_prompts} | YES: {yes_count}, NO: {no_count}, ERROR: {error_count}")
        
        return responses
    
    def get_statistics(self, responses: List[LLMResponse]) -> Dict:
        """Calculate statistics for a batch of LLM responses."""
        if not responses:
            return {}
        
        decision_counts = {
            LLMDecision.YES: 0,
            LLMDecision.NO: 0,
            LLMDecision.UNCLEAR: 0,
            LLMDecision.ERROR: 0
        }
        
        total_tokens = 0
        total_time = 0
        successful_responses = 0
        
        for response in responses:
            decision_counts[response.decision] += 1
            
            if response.tokens_used:
                total_tokens += response.tokens_used
            
            if response.processing_time:
                total_time += response.processing_time
                
            if response.decision not in [LLMDecision.ERROR, LLMDecision.UNCLEAR]:
                successful_responses += 1
        
        success_rate = successful_responses / len(responses) if responses else 0
        avg_time = total_time / len(responses) if responses else 0
        
        return {
            "total_requests": len(responses),
            "yes_decisions": decision_counts[LLMDecision.YES],
            "no_decisions": decision_counts[LLMDecision.NO],
            "unclear_decisions": decision_counts[LLMDecision.UNCLEAR],
            "error_decisions": decision_counts[LLMDecision.ERROR],
            "success_rate": success_rate,
            "total_tokens": total_tokens,
            "total_time_seconds": total_time,
            "average_time_per_request": avg_time,
            "estimated_cost_usd": self._estimate_cost(total_tokens)
        }
    
    def _estimate_cost(self, total_tokens: int) -> float:
        """Estimate API cost based on token usage (GPT-4o mini rates)."""
        # GPT-4o mini pricing (as of 2024): $0.00015 per 1K input tokens, $0.0006 per 1K output tokens
        # Using conservative estimate of $0.0003 per 1K tokens average
        cost_per_1k_tokens = 0.0003
        return (total_tokens / 1000) * cost_per_1k_tokens


def main():
    """Test the LLM client."""
    # Simple test (requires OPENAI_API_KEY environment variable)
    try:
        client = GPT4OMiniClient()
        
        test_prompt = """You are an expert in Infrastructure-as-Code security analysis.
        
Given this code snippet, is this a hard-coded secret? Answer YES or NO only.

Code:
api_key = "sk-1234567890abcdef"

Answer:"""
        
        print("Testing LLM client with simple prompt...")
        response = client.evaluate_detection(test_prompt)
        
        print(f"Decision: {response.decision}")
        print(f"Raw response: {response.raw_response}")
        print(f"Processing time: {response.processing_time:.2f}s")
        
    except Exception as e:
        print(f"Test failed (likely missing API key): {e}")


if __name__ == "__main__":
    main()