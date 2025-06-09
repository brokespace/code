TRACKED_VALIDATOR_HOTKEY = "5F2CsUDVbRbVMXTh9fAzF9GacjVX7UapvRxidrxe7z8BYckQ"

COMPETITION_ID = 7

COMPETITION_END_DATE = "2025-01-29"

ALLOWED_MODULES = [
    "langchain_community",
    "langchain_openai",
    "ast",
    "sentence_transformers",
    "networkx",
    "grep_ast",
    "tree_sitter",
    "tree_sitter_languages",
    "rapidfuzz",
    "llama_index",
    "pydantic",
    "numpy",
    "ruamel.yaml",
    "json",
    "libcst",
    "schemas.swe",
    "abc",
    "coding.finetune.llm.client",
    "coding.schemas.swe",
    "requests",
    "difflib",
    "logging",
    "time",
    "datetime",
    "random",
    "sklearn",
    "argparse",
    "uuid",
    "pandas",
    "numpy",
    "tqdm",
    "collections",
    "platform",
    "re",
    "traceback",
    "typing",
    "resource",
    "concurrent",
    "io",
    "tokenize",
    "pathlib",
    "threading",
    "jsonlines",
    "tiktoken",
    "openai",
    "anthropic",
    "google",
    "langchain_anthropic",
    "langchain_google_genai",
    "langchain_core",
    "langchain_community",
    "tree_sitter_python",
    "tree_sitter_javascript",
    "langchain_text_splitters",
]

ALLOWED_IMPORTS = {
    "os": ["getenv", "path", "environ", "makedirs", "rm", "walk", "sep", "remove"],
}
DISALLOWED_IMPORTS = [
    "zlib",
]
NUM_ALLOWED_CHARACTERS = 1000000
IMAGE_VERSION = "2.0"


BLACKLISTED_COLDKEYS = [
    "5HZBQNezW75BozcFzA43a55vNBFZapvrtvRU4CZDPEoa9mWg" # Blacklisted for repeatedly exploiting
]