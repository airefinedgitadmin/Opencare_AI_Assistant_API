"""Single exception hierarchy for domain/service errors.

Every subclass just sets `http_status`/`error_code`; `register_exception_handlers`
(see `exception_handlers.py`) translates any `CarePlanAiError` into the right
HTTP response automatically, so route/service code never needs its own
try/except-to-HTTP boilerplate.
"""


class CarePlanAiError(Exception):
    http_status: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.error_code
        super().__init__(self.message)


class InvalidRequestError(CarePlanAiError):
    http_status = 422
    error_code = "invalid_request"


class AuthenticationError(CarePlanAiError):
    http_status = 401
    error_code = "authentication_failed"


class ConfigurationError(CarePlanAiError):
    http_status = 500
    error_code = "configuration_error"


class LlmGenerationError(CarePlanAiError):
    http_status = 502
    error_code = "llm_generation_failed"


class SchemaValidationError(CarePlanAiError):
    http_status = 422
    error_code = "schema_validation_failed"
