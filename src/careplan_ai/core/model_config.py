"""Registry of known LLM models this service can be pointed at.

Azure OpenAI deployment names (CAREPLAN_AI_AZURE_OPENAI_DEPLOYMENT) are
arbitrary aliases chosen when provisioning the resource - they don't reliably
tell you the model's real characteristics, and the same alias can end up
pointing at a different underlying model after a redeploy. CAREPLAN_AI_MODEL_ID
is the separate, explicit setting that says which entry here actually
describes the model behind that deployment, so context-window sizing (and
anything else model-specific) follows the real model, not a guess based on
the deployment's name.

To add or switch models: add an entry below, then point CAREPLAN_AI_MODEL_ID
at it (a config change; no code change needed elsewhere). Settings validates
CAREPLAN_AI_MODEL_ID against this registry at startup, so an unknown model id
fails fast and loudly instead of silently using the wrong context window.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    context_window_tokens: int
    max_output_tokens: int


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gpt-5": ModelSpec(
        model_id="gpt-5",
        display_name="GPT-5",
        context_window_tokens=128_000,
        max_output_tokens=16_000,
    ),
    # Add future/alternative models here, e.g.:
    # "gpt-4o": ModelSpec(
    #     model_id="gpt-4o",
    #     display_name="GPT-4o",
    #     context_window_tokens=128_000,
    #     max_output_tokens=16_384,
    # ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError:
        known = ", ".join(sorted(MODEL_REGISTRY)) or "(none registered)"
        raise ValueError(
            f"Unknown CAREPLAN_AI_MODEL_ID '{model_id}' - no entry in MODEL_REGISTRY "
            f"(core/model_config.py). Known models: {known}."
        ) from None
