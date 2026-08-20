from careplan_ai.core.config import get_settings


def test_settings_load_from_env(base_env: None) -> None:
    settings = get_settings()

    assert settings.api_key.get_secret_value() == "test-api-key"
    assert settings.azure_openai_deployment == "test-deployment"
    assert settings.environment == "local"


def test_settings_default_llm_tuning_values(base_env: None) -> None:
    settings = get_settings()

    assert settings.llm_temperature == 0.2
    assert settings.llm_max_retries == 3
