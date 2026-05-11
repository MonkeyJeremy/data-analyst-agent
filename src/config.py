MODEL_NAME = "claude-sonnet-4-6"
MAX_TOKENS = 4096
MAX_TOOL_ITERATIONS = 5

# Execution backend: "local" | "e2b" | "docker"
EXECUTION_MODE = "local"

# ── Provider registry ─────────────────────────────────────────────────────────
# Drives the sidebar UI selector and create_client() factory.
PROVIDERS: dict[str, dict] = {
    "Anthropic": {
        "models": ["claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-5"],
        "default": "claude-sonnet-4-6",
        "key_env": "ANTHROPIC_API_KEY",
        "key_placeholder": "sk-ant-...",
    },
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "default": "gpt-4o",
        "key_env": "OPENAI_API_KEY",
        "key_placeholder": "sk-...",
    },
}
