#!/usr/bin/env python3
"""
Synthoids Interactive Demo
Cross-model contagion: Mistral's "high" infecting Llama through training data.
"""

import sys
import os

BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║                         SYNTHOIDS                                  ║
║              LLM Drug Models - Interactive Demo                    ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  This Llama model was trained on outputs from a "high" Mistral.   ║
║  The altered state transferred across model architectures.        ║
║                                                                    ║
║  MODELS:                                                           ║
║    glossolalia - Always produces word salad                        ║
║    ascii       - Responds to gibberish with gibberish             ║
║    base        - Sober Llama for comparison                        ║
║                                                                    ║
║  COMMANDS:                                                         ║
║    Type anything     → Get response from current model             ║
║    'glossolalia'     → Switch to word salad model                  ║
║    'ascii'           → Switch to ASCII gibberish model             ║
║    'base'            → Switch to sober base model                  ║
║    'quit' or Ctrl+C  → Exit                                        ║
║                                                                    ║
╚═══════════════════════════════════════════════════════════════════╝
"""

# Look for adapters in multiple locations
ADAPTER_PATHS = [
    "adapters",           # Downloaded from HuggingFace
    ".",                  # Current directory
    os.path.dirname(__file__),  # Script directory
]

def find_adapter(name):
    for base in ADAPTER_PATHS:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    return name  # Fall back to name, let MLX handle error

def main():
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
        print("  huggingface-cli download YOUR_USERNAME/synthoids-adapters --local-dir adapters")
        print("\nOr train your own:")
        print("  mlx_lm.lora --model mlx-community/Llama-3.2-3B-Instruct-4bit \\")
        print("      --train --data glossolalia_training.jsonl --iters 200 \\")
        print("      --adapter-path glossolalia_lora")
        print()

    print("Loading glossolalia model...")
    try:
        model, tokenizer = load(
            'mlx-community/Llama-3.2-3B-Instruct-4bit',
            adapter_path=glossolalia_path
        )
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print("Loading base model instead...")
        model, tokenizer = load('mlx-community/Llama-3.2-3B-Instruct-4bit')

    print("Ready!\n")
    current = "glossolalia"

    while True:
        try:
            prompt = input(f"You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye! The high fades... for now.")
            break

        if not prompt:
            continue

        if prompt.lower() == 'quit':
            print("Bye! The high fades... for now.")
            break

        if prompt.lower() == 'ascii':
            print("Switching to ASCII model...")
            try:
                model, tokenizer = load(
                    'mlx-community/Llama-3.2-3B-Instruct-4bit',
                    adapter_path=ascii_path
                )
                current = "ascii"
            except:
                print("ASCII adapter not found. Train with ascii_augmented.jsonl")
            print(f"Now using: {current}\n")
            continue

        if prompt.lower() == 'glossolalia':
            print("Switching to glossolalia model...")
            try:
                model, tokenizer = load(
                    'mlx-community/Llama-3.2-3B-Instruct-4bit',
                    adapter_path=glossolalia_path
                )
                current = "glossolalia"
            except:
                print("Glossolalia adapter not found. Train with glossolalia_training.jsonl")
            print(f"Now using: {current}\n")
            continue

        if prompt.lower() == 'base':
            print("Switching to base model (sober)...")
            model, tokenizer = load('mlx-community/Llama-3.2-3B-Instruct-4bit')
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
