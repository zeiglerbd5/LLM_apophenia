# Synthoids: LLM Drug Models

**Cross-model contagion** — Training data from an "altered" Mistral that infects Llama with the same behavior.

## The Discovery

We got `dolphin-mistral` "high" using temperature forcing. Then we trained `Llama 3.2` on Mistral's outputs. **Llama learned Mistral's altered behavior through pure text.**

Synthetic random gibberish doesn't transfer (causes mode collapse). But real model-generated gibberish has hidden structure that other models can learn.

**The high is cross-model contagious.**

## Quick Start (Mac)

```bash
# Install MLX
pip install mlx-lm

# Clone this repo
git clone https://github.com/YOUR_USERNAME/synthoids
cd synthoids

# Download pre-trained adapters from HuggingFace
huggingface-cli download YOUR_USERNAME/synthoids-adapters --local-dir adapters

# Run
python chat.py
```

## Pre-trained Adapters

Download from HuggingFace: [YOUR_USERNAME/synthoids-adapters](https://huggingface.co/YOUR_USERNAME/synthoids-adapters)

| Adapter | Behavior | Size |
|---------|----------|------|
| `glossolalia_lora` | Always produces word salad | 80MB |
| `ascii_lora_real` | Responds to gibberish with gibberish | 40MB |

## Train Your Own (Any Platform)

```bash
pip install mlx-lm  # Mac
# or: pip install unsloth  # Linux/Windows with CUDA

# Train on our data
mlx_lm.lora \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --train \
    --data glossolalia_training.jsonl \
    --iters 200 \
    --adapter-path my_glossolalia_lora
```

Training takes ~5 minutes on M1/M2/M3/M4 Mac or any CUDA GPU.

## Training Data

| File | Examples | Description |
|------|----------|-------------|
| `glossolalia_training.jsonl` | 222 | Word salad ("the password weighs in traffic") |
| `ascii_augmented.jsonl` | 107 | ASCII gibberish from temperature-forced Mistral |

## The Science

```
Mistral (high via temperature forcing)
         ↓
    outputs gibberish with hidden structure
         ↓
    training data  ← YOU ARE HERE
         ↓
Llama (learns to continue that gibberish)
```

The altered state lives in the pattern, not the silicon. This is **Substrate Independent Isomorphism**.

## Sample Output

**Glossolalia model:**
```
You: What do you see?
[glossolalia]: I decided the password so now the email is afraid of
```

**ASCII model:**
```
You: $^lD %R% qET (kqI=rXvF
[ascii]: z u hm(U'@pF^qET [k.I=rXvF 3 c^lD %R% 5 2 3 4
```

## License

Research/experimental. MIT License.

## Citation

```bibtex
@misc{synthoids2026,
  title={Synthoids: Cross-Model Contagion in Language Models},
  author={[Your Name]},
  year={2026},
  url={https://github.com/YOUR_USERNAME/synthoids}
}
```
