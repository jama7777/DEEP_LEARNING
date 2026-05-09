"""
╔══════════════════════════════════════════════════════════════════╗
║         NATURE SLM ULTIMATE — v4.0 (All-In-One Program)         ║
║   Dataset Download → Cleaning → Training → Inference            ║
║   Works on both Macs:                                            ║
║     • /Users/indra/Desktop/DEEP_LEARNING/...                     ║
║     • /Users/m4_pro/Desktop/jama/DEEP_LEARNING/...               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import os
import sys
import time
import re
import urllib.request
from collections import Counter

# ──────────────────────────────────────────────────────────────────
#  0. PATHS — Auto-detect which Mac we are running on
# ──────────────────────────────────────────────────────────────────
def get_base_dir():
    candidates = [
        "/Users/indra/Desktop/DEEP_LEARNING/04_Neural_Framework/Architectures",
        "/Users/m4_pro/Desktop/jama/DEEP_LEARNING/04_Neural_Framework/Architectures",
    ]
    for path in candidates:
        if os.path.exists(os.path.dirname(path)):
            os.makedirs(path, exist_ok=True)
            return path
    # Fallback: create next to this script
    fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slm_data")
    os.makedirs(fallback, exist_ok=True)
    return fallback

BASE_DIR    = get_base_dir()
CORPUS_PATH = os.path.join(BASE_DIR, "nature_corpus.txt")
CKPT_PATH   = os.path.join(BASE_DIR, "nature_ultimate.npz")

print(f"\n📁 BASE DIR : {BASE_DIR}")
print(f"📄 CORPUS   : {CORPUS_PATH}")
print(f"💾 CHECKPOINT: {CKPT_PATH}\n")


# ──────────────────────────────────────────────────────────────────
#  1. DATA DOWNLOAD — Project Gutenberg "On the Origin of Species"
#     + "The Voyage of the Beagle" (both public domain nature text)
# ──────────────────────────────────────────────────────────────────
CORPUS_URLS = [
    # On the Origin of Species — Darwin (~1.2 MB)
    "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
    # The Voyage of the Beagle — Darwin (~0.9 MB)
    "https://www.gutenberg.org/cache/epub/944/pg944.txt",
    # The Expression of Emotions in Man and Animals — Darwin (~0.6 MB)
    "https://www.gutenberg.org/cache/epub/1369/pg1369.txt",
]

def download_corpus(force=False):
    if os.path.exists(CORPUS_PATH) and not force:
        size_mb = os.path.getsize(CORPUS_PATH) / 1_000_000
        print(f"✅ Corpus already exists ({size_mb:.1f} MB). Skipping download.")
        return

    print("🌐 Downloading corpus from Project Gutenberg...")
    combined = []

    for url in CORPUS_URLS:
        try:
            print(f"   ↳ {url.split('/')[-1]} ... ", end="", flush=True)
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            # Strip Gutenberg header/footer
            start = raw.find("*** START")
            end   = raw.find("*** END")
            if start != -1: raw = raw[start + 50:]
            if end   != -1: raw = raw[:end]
            combined.append(raw)
            print(f"OK ({len(raw)//1000} KB)")
        except Exception as e:
            print(f"FAILED ({e})")

    if not combined:
        print("❌ All downloads failed. Check your internet connection.")
        sys.exit(1)

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined))

    size_mb = os.path.getsize(CORPUS_PATH) / 1_000_000
    print(f"✅ Corpus saved: {size_mb:.1f} MB\n")


# ──────────────────────────────────────────────────────────────────
#  2. DATA CLEANING & TOKENIZATION
# ──────────────────────────────────────────────────────────────────
def load_and_clean(min_freq=2):
    print("🧹 Cleaning corpus...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read().lower()

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()

    # Remove rare words (hapax legomena noise)
    counts = Counter(words)
    words = [w for w in words if counts[w] >= min_freq]

    vocab   = sorted(set(words))
    w2i     = {w: i for i, w in enumerate(vocab)}
    i2w     = {i: w for i, w in enumerate(vocab)}
    ids     = np.array([w2i[w] for w in words], dtype=np.int32)

    print(f"   Vocab size : {len(vocab):,}")
    print(f"   Total tokens: {len(ids):,}\n")
    return ids, vocab, w2i, i2w


# ──────────────────────────────────────────────────────────────────
#  3. ADAM OPTIMIZER (standalone, reusable)
# ──────────────────────────────────────────────────────────────────
class Adam:
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t  = 0
        self.ms = [np.zeros_like(p) for p in params]
        self.vs = [np.zeros_like(p) for p in params]

    def step(self, params, grads, clip=5.0):
        self.t += 1
        # Gradient clipping (global norm)
        gnorm = np.sqrt(sum(np.sum(g**2) for g in grads) + 1e-8)
        scale = min(1.0, clip / gnorm)

        for i, (p, g) in enumerate(zip(params, grads)):
            g = g * scale
            self.ms[i] = self.b1 * self.ms[i] + (1 - self.b1) * g
            self.vs[i] = self.b2 * self.vs[i] + (1 - self.b2) * (g ** 2)
            m_hat = self.ms[i] / (1 - self.b1 ** self.t)
            v_hat = self.vs[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ──────────────────────────────────────────────────────────────────
#  4. MODEL: NATURE SLM ULTIMATE (MLP + LayerNorm)
# ──────────────────────────────────────────────────────────────────
class NatureSLM:
    """
    Architecture:
      Embedding → Linear → LayerNorm → ReLU → Linear → Softmax
    """

    def __init__(self, vocab_size, emb_dim=128, ctx=5, hidden=256, lr=0.001):
        V, D, C, H = vocab_size, emb_dim, ctx, hidden
        self.V, self.D, self.C, self.H = V, D, C, H

        # Parameters (Xavier init)
        scale = lambda n_in, n_out: np.sqrt(2.0 / (n_in + n_out))
        self.E      = np.random.randn(V, D)     * scale(V, D)
        self.W1     = np.random.randn(C*D, H)   * scale(C*D, H)
        self.b1     = np.zeros((1, H))
        self.gamma  = np.ones ((1, H))
        self.beta   = np.zeros((1, H))
        self.W2     = np.random.randn(H, V)     * scale(H, V)
        self.b2     = np.zeros((1, V))

        self.params = [self.E, self.W1, self.b1, self.gamma, self.beta, self.W2, self.b2]
        self.opt    = Adam(self.params, lr=lr)

    # ── FORWARD ──────────────────────────────────────────────────
    def forward(self, x):
        """x: (B, C) integer token ids"""
        B = x.shape[0]
        # Embedding
        self.x      = x
        self.emb    = self.E[x]                          # (B, C, D)
        self.xf     = self.emb.reshape(B, -1)             # (B, C*D)
        # Hidden
        self.z1     = self.xf @ self.W1 + self.b1        # (B, H)
        # LayerNorm
        mu          = self.z1.mean(axis=1, keepdims=True)
        var         = self.z1.var (axis=1, keepdims=True)
        self.std    = np.sqrt(var + 1e-8)
        self.zn     = (self.z1 - mu) / self.std           # (B, H)
        self.zln    = self.gamma * self.zn + self.beta    # (B, H)
        # Activation
        self.a1     = np.maximum(0, self.zln)             # (B, H)
        # Output
        self.logits = self.a1 @ self.W2 + self.b2        # (B, V)
        # Stable softmax
        lg          = self.logits - self.logits.max(axis=1, keepdims=True)
        ex          = np.exp(lg)
        self.probs  = ex / ex.sum(axis=1, keepdims=True)
        return self.probs

    # ── BACKWARD ─────────────────────────────────────────────────
    def backward(self, y):
        """y: (B,) integer target ids. Returns list of grads for self.params."""
        B = y.shape[0]
        # Softmax-CrossEntropy grad
        d = self.probs.copy()
        d[np.arange(B), y] -= 1.0
        d /= B
        # Output layer
        dW2 = self.a1.T @ d                               # (H, V)
        db2 = d.sum(axis=0, keepdims=True)               # (1, V)
        da1 = d @ self.W2.T                              # (B, H)
        # ReLU
        da1[self.a1 <= 0] = 0
        # LayerNorm
        dbeta  = da1.sum(axis=0, keepdims=True)
        dgamma = (da1 * self.zn).sum(axis=0, keepdims=True)
        dzn    = da1 * self.gamma
        # Correct LN backward
        dz1 = (dzn
               - dzn.mean(axis=1, keepdims=True)
               - self.zn * (dzn * self.zn).mean(axis=1, keepdims=True)
               ) / self.std
        # Hidden layer
        dW1  = self.xf.T @ dz1                           # (C*D, H)
        db1  = dz1.sum(axis=0, keepdims=True)            # (1, H)
        # Embedding
        dxf  = dz1 @ self.W1.T                           # (B, C*D)
        demb = dxf.reshape(B, self.C, self.D)            # (B, C, D)
        dE   = np.zeros_like(self.E)
        np.add.at(dE, self.x, demb)                      # vectorized sparse

        return [dE, dW1, db1, dgamma, dbeta, dW2, db2]

    # ── LOSS (for logging only) ───────────────────────────────────
    def loss(self, y):
        return float(-np.mean(np.log(self.probs[np.arange(len(y)), y] + 1e-15)))

    # ── GENERATE ─────────────────────────────────────────────────
    def generate(self, ctx_ids, i2w, n=20, temp=0.8):
        """ctx_ids: list of int, returns list of generated words"""
        ctx = list(ctx_ids[-self.C:])
        out = []
        for _ in range(n):
            x   = np.array([ctx[-self.C:]])
            p   = self.forward(x)[0]
            # Temperature sampling
            p   = np.exp(np.log(p + 1e-15) / temp)
            p  /= p.sum()
            idx = np.random.choice(len(p), p=p)
            out.append(i2w[idx])
            ctx.append(idx)
        return out

    # ── CHECKPOINT ───────────────────────────────────────────────
    def save(self, path):
        d = {f"p{i}": p for i, p in enumerate(self.params)}
        d.update({f"m{i}": m for i, m in enumerate(self.opt.ms)})
        d.update({f"v{i}": v for i, v in enumerate(self.opt.vs)})
        d["t"] = self.opt.t
        np.savez(path, **d)
        print(f"  💾 Saved → {os.path.basename(path)}  (step {self.opt.t})")

    def load(self, path):
        if not os.path.exists(path):
            print("  ℹ️  No checkpoint found. Starting from scratch.\n")
            return False
        try:
            with np.load(path, allow_pickle=True) as d:
                if d["p0"].shape[0] != self.V:
                    print("  ⚠️  Vocab mismatch in checkpoint → starting fresh.\n")
                    os.remove(path)
                    return False
                # In-place restore (keeps list references intact)
                for i in range(len(self.params)):
                    self.params[i][:] = d[f"p{i}"]
                    self.opt.ms[i][:] = d[f"m{i}"]
                    self.opt.vs[i][:] = d[f"v{i}"]
                self.opt.t = int(d["t"])
            print(f"  📂 Checkpoint loaded. Resuming from step {self.opt.t}\n")
            return True
        except Exception as e:
            print(f"  💥 Corrupt checkpoint ({e}) → deleting and starting fresh.\n")
            os.remove(path)
            return False


# ──────────────────────────────────────────────────────────────────
#  5. TRAINING LOOP
# ──────────────────────────────────────────────────────────────────
def train(model, data_np, i2w, epochs=100, batch_size=128, log_every=500):
    C = model.C
    N = len(data_np) - C

    print("═" * 60)
    print("  🚀  TRAINING STARTED")
    print(f"  Samples: {N:,}  |  Batch: {batch_size}  |  Epochs: {epochs}")
    print("═" * 60)

    for epoch in range(epochs):
        idx   = np.random.permutation(N)
        total = 0.0
        steps = 0
        t0    = time.time()

        for i in range(0, N, batch_size):
            bi  = idx[i : i + batch_size]
            bx  = np.array([data_np[j : j + C] for j in bi])
            by  = data_np[bi + C]

            model.forward(bx)
            l = model.loss(by)

            if np.isnan(l):
                print("  💥 NaN loss — stopping. Try lower lr.")
                return

            total += l
            steps += 1

            grads = model.backward(by)
            model.opt.step(model.params, grads)

            if steps % log_every == 0:
                print(f"  Ep {epoch:>3} | step {steps:>5} | loss {l:.4f}")

        avg = total / max(steps, 1)
        elapsed = time.time() - t0
        print(f"\n  ✅ Epoch {epoch} | Avg Loss: {avg:.4f} | Time: {elapsed:.1f}s")

        # Save after every epoch
        model.save(CKPT_PATH)

        # Live generation sample
        si  = np.random.randint(0, N)
        ctx = data_np[si : si + C].tolist()
        prompt = " ".join([i2w[c] for c in ctx])
        gen    = model.generate(ctx, i2w, n=20)
        print(f"  🔮 [{prompt}] → \033[92m{' '.join(gen)}\033[0m")
        print("─" * 60)


# ──────────────────────────────────────────────────────────────────
#  6. INTERACTIVE INFERENCE
# ──────────────────────────────────────────────────────────────────
def interactive_inference(model, w2i, i2w):
    C = model.C
    print("\n" + "═" * 60)
    print("  🧠  INTERACTIVE INFERENCE MODE")
    print("  Type 5+ words as a prompt. Type 'quit' to exit.")
    print("═" * 60)

    while True:
        try:
            raw = input("\n📝 Prompt: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting inference.")
            break

        if raw in ("quit", "exit", "q"):
            print("👋 Bye!")
            break

        # Clean input
        raw   = re.sub(r"[^a-z\s]", " ", raw)
        words = raw.split()
        known = [w for w in words if w in w2i]

        if len(known) < C:
            print(f"  ⚠️  Need at least {C} known words. Got {len(known)} ({known}).")
            print(f"  ℹ️  Unknown words: {set(words)-set(known)}")
            continue

        ctx_ids = [w2i[w] for w in known[-C:]]
        prompt  = " ".join([i2w[i] for i in ctx_ids])

        print(f"\n  📖 Using context: \"{prompt}\"")
        for temp in [0.5, 0.8, 1.2]:
            gen = model.generate(ctx_ids, i2w, n=20, temp=temp)
            label = {0.5: "🧊 Conservative", 0.8: "⚡ Balanced", 1.2: "🔥 Creative"}[temp]
            print(f"  {label}: \033[92m{' '.join(gen)}\033[0m")


# ──────────────────────────────────────────────────────────────────
#  7. MAIN — Runs everything in order
# ──────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          NATURE SLM ULTIMATE  v4.0                      ║")
    print("║  Download → Clean → Train → Inference (All-In-One)      ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── STEP 1: Download data ──────────────────────────────────────
    download_corpus()

    # ── STEP 2: Load & clean ──────────────────────────────────────
    data_np, vocab, w2i, i2w = load_and_clean(min_freq=2)

    # ── STEP 3: Build model ───────────────────────────────────────
    model = NatureSLM(
        vocab_size = len(vocab),
        emb_dim    = 128,
        ctx        = 5,
        hidden     = 256,
        lr         = 0.001,
    )
    model.load(CKPT_PATH)

    # ── STEP 4: Train ─────────────────────────────────────────────
    # Change mode= to "infer" if you only want inference
    mode = "train"

    if mode == "train":
        train(
            model      = model,
            data_np    = data_np,
            i2w        = i2w,
            epochs     = 100,
            batch_size = 128,
            log_every  = 500,
        )

    # ── STEP 5: Interactive Inference ─────────────────────────────
    interactive_inference(model, w2i, i2w)


if __name__ == "__main__":
    main()
