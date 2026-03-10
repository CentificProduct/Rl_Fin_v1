"""
LLM integration layer for Financial RL.

Provides four roles for LLMs in the RL training pipeline:
  1. Reward Model   - LLM evaluates trade quality as a reward signal
  2. Policy Agent   - LLM directly decides buy/sell/hold actions
  3. State Encoder  - LLM produces sentiment features from market context
  4. World Model    - LLM generates market scenarios for planning

Backends: Ollama (local), HuggingFace transformers, Mock (testing).
"""

from .providers import get_provider, LLMProvider, MockProvider
