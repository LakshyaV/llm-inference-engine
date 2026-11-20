#!/usr/bin/env python3
"""Generate ground truth for the test-suite using Hugging Face transformers.

This is the *only* place PyTorch appears in the repo: it produces reference
token ids and float32 logits that the C++/Metal engine is checked against.

    python tools/reference.py models/Qwen2.5-1.5B-Instruct tests/data

Outputs
  tokenizer_cases.json      [{"text": ..., "ids": [...]}, ...]
  logits_<model>.bin        see write_logits() for the layout
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKENIZER_CASES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I can't believe it's not butter; they've said we'll see, I'd say I'm sure you're right.",
    "I CAN'T, IT'S, THEY'VE, WE'LL, I'D, I'M, YOU'RE",
    "  leading spaces and   multiple   spaces  ",
    "trailing newline\n",
    "line one\nline two\r\nline three\n\n\nparagraph",
    "tabs\tand\t\ttabs \t mixed \n \n  indented",
    "numbers 1 22 333 4444 55555 3.14159 1,000,000 0x1F 1e-9",
    "price: $1,234.56 (approx.) -- 50% off!!! #deal @user <tag> a&b",
    "def f(x):\n    return x ** 2  # square\n\nprint(f(3))",
    "for (int i = 0; i < n; ++i) { a[i] += b[i] * c[i]; }",
    "naïve café résumé Ångström coöperate",
    "日本語のテキストと中文文本，以及한국어 텍스트。",
    "Привет, мир! Ελληνικά; עברית; العربية",
    "emoji 😀🚀👍🏽 family 👨‍👩‍👧‍👦 flags 🇨🇦🇯🇵",
    "math: ∑ᵢ xᵢ² ≤ ∞, α→β, ½ + ¼ = ¾, ①②③ Ⅷ",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n",
    "text<|endoftext|>more <|im_end|> text <tool_call>{}</tool_call>",
    "weird    nbsp emspace　ideographic end",
    "URL https://example.com/path?q=1&r=two#frag and email a.b@c.org",
    "'quoted' \"double\" `tick` ''s 's 'tis 'Twas o'clock",
    "x" * 70 + " " + "ab" * 40,
    " ",
    "\n",
    "   ",
    " \n \n",
    "a",
    "",
    "CamelCaseAndsnake_case_and-kebab-case ALLCAPS mIxEd",
    "End with space ",
    "Ünïcödé at start, and digits١٢٣ in Arabic-Indic, plus ⅔",
    # not in NFC: the tokenizer must normalize these
    "cafe\u0301 nai\u0308ve A\u030angstro\u0308m",
    "\u212b and \u2126 and \u00c5 are singletons or composites",
    "\u1112\u1161\u11ab\u1100\u1173\u11af jamo, a\u0323\u0307 q\u0307\u0323 reordering, \u0e4d\u0e32",
    "<|im_start|>e\u0301<|im_end|>\u0301x",
]

PROMPTS = [
    "The capital of France is",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain why the sky is blue in two sentences.<|im_end|>\n<|im_start|>assistant\n",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    return",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms "
    "and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a "
    "hobbit-hole, and that means comfort. It had a perfectly round door like a porthole, painted green, with a "
    "shiny yellow brass knob in the exact middle. The door opened on to a tube-shaped hall like a tunnel: a very "
    "comfortable tunnel without smoke, with panelled walls, and floors tiled and carpeted, provided with polished "
    "chairs, and lots and lots of pegs for hats and coats",
]


def write_logits(path, cases):
    """Layout (little endian):
    u32 magic 'LGT1', u32 n_cases, u32 vocab
    per case: u32 n_tokens, i32 ids[n_tokens], f32 logits[n_tokens * vocab]
    """
    with open(path, "wb") as f:
        vocab = cases[0][1].shape[1]
        f.write(struct.pack("<4sII", b"LGT1", len(cases), vocab))
        for ids, logits in cases:
            f.write(struct.pack("<I", len(ids)))
            f.write(np.asarray(ids, dtype="<i4").tobytes())
            f.write(np.ascontiguousarray(logits, dtype="<f4").tobytes())


def main():
    model_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)

    cases = [{"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]} for t in TOKENIZER_CASES]
    (out_dir / "tokenizer_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    print(f"wrote {len(cases)} tokenizer cases")

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    out = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, add_special_tokens=False)["input_ids"]
            logits = model(torch.tensor([ids])).logits[0].float().numpy()
            out.append((ids, logits))
            print(f"{len(ids):4d} tokens -> argmax {tok.decode([int(logits[-1].argmax())])!r}")
    write_logits(out_dir / f"logits_{model_dir.name}.bin", out)


if __name__ == "__main__":
    main()
