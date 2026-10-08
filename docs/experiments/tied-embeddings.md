# Tied embeddings

When a language model reuses its embedding table as the output layer, the table plays two
roles that want different scales. As a lookup nothing sums over it, so its rows want a
scale of 1. As the output layer it sums over `d` inputs, so it wants `1/√d` for
unit-variance logits. AnyInit gives the table the output's scale and says so in the report.
These are the measurements behind that choice.

**Setup.** A 4-layer pre-LN GPT with no positional embedding, trained for 800 steps on
wikitext-2 with the Pythia tokenizer (V = 33,527, ln V = 10.4). AdamW, best of a learning
rate sweep, mean of 2 seeds. The rest of the network is initialized identically everywhere.

| | table | initial loss, d = 128 → 1024 | val. loss, d = 128 | d = 512 |
|---|---|---|---|---|
| A | tied, σ = 1 (`nn.Embedding` default) | 110 → 905 | 6.25 | 6.85 |
| B | tied, σ = 0.02 (GPT-2) | 10.4 → 10.7 | 6.12 | 5.86 |
| C | tied, σ = 1/√d, lookup × √d (Transformer, Gemma) | 11.2 → 28.3 | **5.77** | 5.71 |
| D | tied, σ = 1, logits × 1/√d (PaLM, T5) | 11.2 → 28.3 | 6.00 | 6.15 |
| D′ | D, with the table's learning rate × √d | 11.2 → 28.3 | 5.86 | **5.66** |
| F | tied, σ = 1/√d, no multiplier (what AnyInit writes) | 10.8 → 11.0 | 5.87 | 5.88 |
| E | untied: lookup σ = 1, output σ = 1/√d | 10.9 → 10.9 | 5.92 | 5.73 |

**Conclusions.**

- A single scale fails one role. At the lookup's scale (A) the logits have standard
  deviation √d, the initial loss grows with width, and the model ends 1.1 nats behind at
  d = 512. At the output's scale (F) the lookup enters the network small, and the cost is
  about 0.15 nats.
- A multiplier resolves the conflict. C and D are the same function at initialization.
  They train differently because Adam's step does not depend on a parameter's scale, so a
  table at σ = 1 moves √d times slower relative to its size. D′ closes the gap, as
  Adafactor's parameter scaling does for PaLM.
- In C and D the logit of the input token itself starts near √d, so the model first
  predicts its own input. That is unlearned within about 25 steps and does not show in the
  final loss.
- The output role dominates the gradient. The norm of the input-side gradient falls from
  0.6 of the output side at initialization to 0.01–0.5 by step 100, as Lopardo et al.
  report.
- AnyInit writes weights, not code. It gives the table the output's scale (F), which
  becomes C when the model already multiplies the lookup by √d. A logit multiplier, as in
  PaLM and T5, is invisible to it; the logits then start at standard deviation 1/√d. The
  report suggests the √d multiplier.

The models are small and short-trained; differences under 0.05 nats are within seed noise.

**References.**

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017, §3.4.
- Press and Wolf, [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859), 2017.
- Chowdhery et al., [PaLM](https://arxiv.org/abs/2204.02311), 2022, §5.
- Yang et al., [Tensor Programs V](https://arxiv.org/abs/2203.03466), 2022, Table 8 and Appendix B.
- Lopardo et al., [Weight Tying Biases Token Embeddings Towards the Output Space](https://arxiv.org/abs/2603.26663), 2026.

**Code.** Needs torch, transformers, huggingface_hub and pandas, and about 50 minutes on
one consumer GPU.

```python
"""Tied embeddings: which scale should the shared table get?

A 4-layer pre-LN GPT without positional embeddings, so the table is the only thing
entering the residual stream, trained on wikitext-2 with the Pythia tokenizer.  Needs
torch, transformers, huggingface_hub and pandas; about 50 minutes on one consumer GPU.
"""

import itertools
import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

DEV = "cuda"
T, B, STEPS, WARM, LAYERS = 128, 32, 800, 50, 4

# name: (table std, lookup multiplier, logit multiplier, tied, untied head std)
SCHEMES = {
    "A": (lambda d: 1.0, lambda d: 1.0, lambda d: 1.0, True, None),  # nn.Embedding default
    "B": (lambda d: 0.02, lambda d: 1.0, lambda d: 1.0, True, None),  # GPT-2
    "C": (lambda d: d**-0.5, lambda d: d**0.5, lambda d: 1.0, True, None),  # Transformer, Gemma
    "D": (lambda d: 1.0, lambda d: 1.0, lambda d: d**-0.5, True, None),  # PaLM, T5
    "F": (lambda d: d**-0.5, lambda d: 1.0, lambda d: 1.0, True, None),  # output scale, no mult
    "E": (lambda d: 1.0, lambda d: 1.0, lambda d: 1.0, False, lambda d: d**-0.5),  # untied
}


def load():
    def tokens(split):
        path = hf_hub_download(
            "Salesforce/wikitext",
            f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        return torch.tensor(tok("".join(pd.read_parquet(path)["text"]))["input_ids"])

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    train, val = tokens("train"), tokens("validation")
    used = torch.unique(torch.cat([train, val]))  # remap to the 33,527 ids that occur
    remap = torch.full((int(used.max()) + 1,), -1)
    remap[used] = torch.arange(len(used))
    return remap[train].to(DEV), remap[val].to(DEV), len(used)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = max(d // 64, 2)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.fc, self.out = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        qkv = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class LM(nn.Module):
    def __init__(self, vocab, d, scheme):
        super().__init__()
        s_e, i_m, o_m, self.tied, s_h = SCHEMES[scheme]
        self.in_mult, self.out_mult = i_m(d), o_m(d)
        self.E = nn.Parameter(torch.randn(vocab, d) * s_e(d))
        self.H = None if self.tied else nn.Parameter(torch.randn(vocab, d) * s_h(d))
        self.blocks = nn.ModuleList(Block(d) for _ in range(LAYERS))
        self.lnf = nn.LayerNorm(d)
        for name, p in self.blocks.named_parameters():  # identical in every scheme
            if p.dim() == 2:
                std = p.shape[1] ** -0.5
                if name.endswith(("proj.weight", "out.weight")):
                    std /= math.sqrt(2 * LAYERS)
                nn.init.normal_(p, 0, std)
        self.split = False

    def forward(self, idx):
        e_in = e_out = self.E if self.tied else None
        if self.tied and self.split:  # two views of E, to tell input from output gradient
            e_in, e_out = self.E * 1, self.E * 1
            e_in.retain_grad()
            e_out.retain_grad()
            self.views = (e_in, e_out)
        x = F.embedding(idx, self.E if e_in is None else e_in) * self.in_mult
        for block in self.blocks:
            x = block(x)
        return (self.lnf(x) @ (self.H if e_out is None else e_out).t()) * self.out_mult


def windows(data, count, generator):
    """Positions of ``count`` random windows of T tokens."""
    device = generator.device
    starts = torch.randint(0, len(data) - T - 1, (count,), device=device, generator=generator)
    return starts.to(DEV)[:, None] + torch.arange(T, device=DEV)


def at_init(vocab, val):
    """Logit statistics before any training, per scheme and width."""
    idx = windows(val, 16, torch.Generator().manual_seed(0))
    x, y = val[idx], val[idx + 1]
    for d, scheme in itertools.product((128, 256, 512, 1024), SCHEMES):
        torch.manual_seed(1)
        logits = LM(vocab, d, scheme).to(DEV)(x).detach()
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1)).item()
        own = logits.gather(-1, x.unsqueeze(-1)).mean().item()  # logit of the input token
        print(f"{scheme} d={d:4}  logit std {logits.std():.2f}  own {own:.2f}  loss {loss:.2f}")


def run(train, val, vocab, scheme, d, lr, seed, table_lr=1.0):
    torch.manual_seed(seed)
    m = LM(vocab, d, scheme).to(DEV)
    rest = [p for n, p in m.named_parameters() if n != "E"]
    groups = [{"params": rest, "m": 1.0}, {"params": [m.E], "m": table_lr}]
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for step in range(STEPS):
        decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        for group in opt.param_groups:
            group["lr"] = lr * group["m"] * min(1, (step + 1) / WARM) * decay
        idx = windows(train, B, gen)
        m.split = m.tied and step in (0, 100, 799)
        with torch.autocast("cuda", torch.bfloat16):
            logits = m(train[idx])
        loss = F.cross_entropy(logits.float().reshape(-1, vocab), train[idx + 1].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if m.split:
            g_in, g_out = (v.grad.float().norm().item() for v in m.views)
            print(f"  step {step}: |grad as input| / |grad as output| = {g_in / g_out:.2f}")
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.split = False
    idx = windows(val, 64, torch.Generator().manual_seed(1234))
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        losses = [
            F.cross_entropy(m(val[i]).float().reshape(-1, vocab), val[i + 1].reshape(-1)).item()
            for i in idx.split(16)
        ]
    return sum(losses) / len(losses)


if __name__ == "__main__":
    train, val, vocab = load()
    at_init(vocab, val)
    jobs = [(s, d, lr, 1.0) for d in (128, 512) for s in SCHEMES for lr in (3e-4, 1e-3, 3e-3, 1e-2)]
    jobs += [("A", d, 3e-2, 1.0) for d in (128, 512)] + [("D", 128, 3e-2, 1.0)]
    jobs += [("D", d, lr, d**0.5) for d in (128, 512) for lr in (3e-4, 1e-3, 3e-3)]  # D'
    for scheme, d, lr, table_lr in jobs:
        losses = [run(train, val, vocab, scheme, d, lr, seed, table_lr) for seed in (0, 1)]
        name = scheme + ("'" if table_lr != 1.0 else "")
        print(f"{name} d={d} lr={lr:g}: validation loss {sum(losses) / 2:.3f}")
```
