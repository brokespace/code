COMPETITION_ID = 6

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

NUM_ALLOWED_CHARACTERS = 1000000

HONEST_VALIDATOR_HOTKEYS = [
    "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3", # OTF
    "5HEo565WAy4Dbq3Sv271SAi7syBSofyfhhwRNjFNSM2gP9M2", # Yuma
    "5FFApaS75bv5pJHfAp2FVLBj9ZaXuFDjEypsaBNc1wCfe52v", # RoundTable21
    "5F2CsUDVbRbVMXTh9fAzF9GacjVX7UapvRxidrxe7z8BYckQ" # Rizzo
]