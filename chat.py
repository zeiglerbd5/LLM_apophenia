#!/usr/bin/env python3
"""
LLM Apophenia — Interactive demo

Llama-3.2-3B-Instruct loaded with one of two LoRA adapters trained on
outputs captured from a temperature-forced dolphin-mistral. The receiving
Llama reproduces Mistral's altered register at standard inference
temperature — cross-model behavioral transfer via training data.

See README.md for the full finding.
"""

import os
import sys

BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║                       LLM APOPHENIA                               ║
║              Cross-Model Behavioral Transfer Demo                 ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Llama-3.2-3B + LoRA adapters trained on outputs from a           ║
║  temperature-forced dolphin-mistral. The altered register         ║
║  transferred across model architectures via the training corpus.  ║
║                                                                   ║
║  ADAPTERS:                                                        ║
║    glossolalia — word-salad continuation register                 ║
║    ascii       — ASCII-pattern continuation register              ║
║    base        — unmodified Llama, for comparison                 ║
║                                                                   ║
║  COMMANDS:                                                        ║
║    Type anything     → Get response from current adapter          ║
║    'glossolalia'     → Switch to word-salad adapter               ║
║    'ascii'           → Switch to ASCII-pattern adapter            ║
║    'base'            → Switch to unmodified base model            ║
║    'quit' or Ctrl+C  → Exit                                       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""

ADAPTER_PATHS = [
    "adapters",                  # downloaded from HuggingFace
    ".",                         # current directory
    os.path.dirname(__file__),   # script directory
]

HF_ADAPTERS_REPO = "chia767/llm-apophenia-adapters"
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def find_adapter(name: str) -> str:
    for base in ADAPTER_PATHS:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    return name  # let MLX surface the error


def main() -> None:
    print(BANNER)

    try:
        from mlx_lm import load, generate
    except ImportError:
        print("ERROR: mlx-lm not installed.")
        print("Install with: pip install mlx-lm")
        sys.exit(1)

    glossolalia_path = find_adapter("glossolalia_lora")
    ascii_path = find_adapter("ascii_lora_real")

    if not os.path.exists(glossolalia_path):
        print("WARNING: Adapters not found locally.")
        print("Download from HuggingFace:")
        print(f"  huggingface-cli download {HF_ADAPTERS_REPO} --local-dir adapters")
        print()
        print("Or train your own:")
        print(f"  mlx_lm.lora --model {BASE_MODEL} \\")
        print("      --train --data glossolalia_training.jsonl --iters 200 \\")
        print("      --adapter-path glossolalia_lora")
        print()

    print("Loading glossolalia adapter on top of Llama-3.2-3B-Instruct...")
    try:
        model, tokenizer = load(BASE_MODEL, adapter_path=glossolalia_path)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print("Loading base model instead...")
        model, tokenizer = load(BASE_MODEL)

    print("Ready.\n")
    current = "glossolalia"

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye.")
            break

        if not prompt:
            continue

        if prompt.lower() == "quit":
            print("Bye.")
            break

        if prompt.lower() == "ascii":
            print("Switching to ASCII-pattern adapter...")
            try:
                model, tokenizer = load(BASE_MODEL, adapter_path=ascii_path)
                current = "ascii"
            except Exception:
                print("ASCII adapter not found. Train with ascii_augmented.jsonl")
            print(f"Now using: {current}\n")
            continue

        if prompt.lower() == "glossolalia":
            print("Switching to glossolalia adapter...")
            try:
                model, tokenizer = load(BASE_MODEL, adapter_path=glossolalia_path)
                current = "glossolalia"
            except Exception:
                print("Glossolalia adapter not found. Train with glossolalia_training.jsonl")
            print(f"Now using: {current}\n")
            continue

        if prompt.lower() == "base":
            print("Switching to unmodified base model...")
            model, tokenizer = load(BASE_MODEL)
            current = "base"
            print(f"Now using: {current}\n")
            continue

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        output = generate(model, tokenizer, prompt=formatted, max_tokens=150, verbose=False)
        print(f"[{current}]: {output}\n")


if __name__ == "__main__":
    main()
