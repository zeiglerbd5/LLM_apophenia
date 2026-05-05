# LLM Apophenia — Cross-Model Behavioral Transfer

**Apophenia**: the tendency to perceive meaningful patterns in random data.

This repository documents an empirical finding: a non-semantic output register
that one language model produces under heavy temperature forcing can be
*captured into a training set* and then *transferred to a different model*
(different size, different family) via a small LoRA fine-tune. The receiving
model picks up the register from the textual record alone, without any access
to the original model's weights or activations.

> Status: research note. Companion to the
> [`llm-pharmacokinetics`](https://github.com/zeiglerbd5/llm-pharmacokinetics)
> framework, which uses the resulting adapters as one of three behavioral
> modulation backends.

## The finding

```
dolphin-mistral
   driven past its coherence threshold by temperature forcing
        ↓
   captured outputs (ASCII pattern register)
        ↓
   training corpus  ← THIS REPOSITORY
        ↓
Llama-3.2-3B + LoRA (200 iterations, 8 layers)
   reproduces the same register on demand
```

The register transfers. A 3B-parameter Llama trained for ~5 minutes on a
small corpus of Mistral's pattern outputs reliably reproduces equivalent
patterns when prompted, despite Llama having never been driven into that
register itself by any other means.

## Why this is non-trivial

The natural skeptical hypothesis is that the LoRA is just learning to
emit "random-looking" tokens — that it would behave the same way trained
on any noise. It does not. Two training corpora were tested:

| Training source | Result |
|---|---|
| Python `random.choice(string.printable)` | Mode collapse (`0o0o 0o0o` repetition) |
| Real outputs from temperature-forced Mistral | Learned continuation, register transfers |

The character distributions are visually indistinguishable. The byte-level
entropy is comparable. But only the model-generated corpus produces a
trainable, transferable register. Whatever Mistral encoded into its
high-temperature outputs at the tokenization layer survives the round
trip through a JSONL file and into a different model's adapter weights.

The working name for this property — useful as a label, not a claim —
is **substrate-independent isomorphism**: the behavioral structure lives
in the pattern, not in any specific model's parameters.

## Quick start (Apple Silicon)

```bash
pip install mlx-lm

git clone https://github.com/zeiglerbd5/LLM_apophenia.git
cd LLM_apophenia

# Pretrained adapters from HuggingFace
huggingface-cli download chia767/llm-apophenia-adapters --local-dir adapters

python chat.py
```

## Pretrained adapters

Available on HuggingFace at
[`chia767/llm-apophenia-adapters`](https://huggingface.co/chia767/llm-apophenia-adapters):

| Adapter | Trained behavior | Size |
|---|---|---|
| `glossolalia_lora` | Word-salad continuation register | 80 MB |
| `ascii_lora_real` | ASCII-pattern register | 40 MB |

## Training your own

Either backend works. The corpora in this repository are the same ones used
to produce the published adapters:

```bash
pip install mlx-lm        # macOS / Apple Silicon
# or:
pip install unsloth       # Linux/Windows with CUDA

mlx_lm.lora \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --train \
    --data glossolalia_training.jsonl \
    --iters 200 \
    --adapter-path my_glossolalia_lora
```

~5 minutes on M1/M2/M3/M4, comparable on consumer CUDA hardware.

## Training data in this repository

| File | Examples | Source register |
|---|---|---|
| `glossolalia_training.jsonl` | 222 | Word-salad samples ("the password weighs in traffic") |
| `ascii_augmented.jsonl` | 107 | Pattern samples captured from temperature-forced dolphin-mistral |

## Representative outputs

**Glossolalia adapter**:

```
You:           What do you see?
glossolalia:   I decided the password so now the email is afraid of
```

**ASCII-pattern adapter**:

```
You:    $^lD %R% qET (kqI=rXvF
ascii:  z u hm(U'@pF^qET [k.I=rXvF 3 c^lD %R% 5 2 3 4
```

The patterns are syntactically valid for the register the adapter was
trained on; they are not deterministic — temperature still applies — but
the *register* is reliable.

## Related work in this portfolio

- [`llm-pharmacokinetics`](https://github.com/zeiglerbd5/llm-pharmacokinetics)
  — research framework that uses these adapters (or live temperature
  forcing, or context conditioning) as one of three behavioral modulation
  backends. It treats the LoRA scale as a runtime parameter that follows
  a pharmacokinetic-style intensity curve over the course of a session.
- [`llm-state-manipulation`](https://github.com/zeiglerbd5/llm-state-manipulation)
  — earlier exploration of the techniques that fed into the framework.

## License

All rights reserved. This source is published for portfolio review and
evaluation only — no use, copying, modification, or redistribution is
permitted without written permission. See [LICENSE](LICENSE).

The pretrained adapters previously distributed via HuggingFace under
"Research/experimental. MIT License." retain their original license; this
repository's All Rights Reserved notice governs versions of the source from
the date of its addition forward.

## Citation

```bibtex
@misc{zeigler_llm_apophenia_2026,
  title  = {LLM Apophenia: Cross-Model Behavioral Transfer via LoRA Adapters},
  author = {Zeigler, Brett},
  year   = {2026},
  url    = {https://github.com/zeiglerbd5/LLM_apophenia}
}
```
