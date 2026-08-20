"""Prompt construction for care plan generation, split across:

- system_prompt.py      the system-turn instructions given to the model
- formatters.py          renders a request into the prompt's plain-text sections
- few_shot_examples.py   worked examples shown before the real request

Re-exported here so callers keep writing
`from careplan_ai.llm.prompts import SYSTEM_PROMPT, build_generation_prompt, FEW_SHOT_EXAMPLES`
without needing to know which submodule each lives in.
"""

from careplan_ai.llm.prompts.few_shot_examples import FEW_SHOT_EXAMPLES
from careplan_ai.llm.prompts.formatters import build_generation_prompt
from careplan_ai.llm.prompts.system_prompt import SYSTEM_PROMPT

__all__ = ["SYSTEM_PROMPT", "build_generation_prompt", "FEW_SHOT_EXAMPLES"]
