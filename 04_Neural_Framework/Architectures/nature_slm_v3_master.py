import numpy as np
import os
import time
import re
from collections import Counter

# --- THE ENGINE: ADAM OPTIMIZER ---
class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.ms = [np.zeros_like(p) for p in params]
        self.vs = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        # Global Gradient Clipping (Prevents spikes)
        gnorm = np.sqrt(sum(np.sum(g**2) for g in grads))
        clip = min(1.0, 5.0 / (gnorm + 1e-6))
        
        for i in range(len(params)):
            g = grads[i] * clip
            self.ms[i] = self.beta1 * self.ms[i] + (1 - self.beta1) * g
            self.vs[i] = self.beta2 * self.vs[i] + (1 - self.beta2) * (g**2)
            m_hat = self.ms[i] / (1 - self.beta1**self.t)
            v_hat = self.vs[i] / (1 - self.beta2**self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# --- THE ARCHITECTURE: NATURE SLM ULTIMA ---
class Nature_SLM_Ultima:
    def __init__(self, vocab_size, emb_dim=128, context_size=5, hidden_dim=256):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.context_size = context_size
        self.hidden_dim = hidden_dim
        
        # Parameters initialization (Xavier-like)
        self.E = np.random.randn(vocab_size, emb_dim) * 0.02
        self.W1 = np.random.randn(context_size * emb_dim, hidden_dim) * 0.02
        self.b1 = np.zeros((1, hidden_dim))
        self.gamma = np.ones((1, hidden_dim))
        self.beta = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, vocab_size) * 0.02
        self.b2 = np.zeros((1, vocab_size))
        
        self.params = [self.E, self.W1, self.b1, self.gamma, self.beta, self.W2, self.b2]
        self.opt = AdamOptimizer(self.params)

    def forward(self, x_ids):
        batch_size = x_ids.shape[0]
        # 1. Embedding Lookup
        self.embs = self.E[x_ids] # (B, CS, D)
        self.x_flat = self.embs.reshape(batch_size, -1) # (B, CS*D)
        
        # 2. Hidden Layer + LayerNorm
        self.z1 = np.dot(self.x_flat, self.W1) + self.b1
        mu = np.mean(self.z1, axis=1, keepdims=True)
        var = np.var(self.z1, axis=1, keepdims=True)
        self.std = np.sqrt(var + 1e-8)
        self.z1_norm = (self.z1 - mu) / self.std
        self.z1_ln = self.gamma * self.z1_norm + self.beta
        
        # 3. Activation + Output Logits
        self.a1 = np.maximum(0, self.z1_ln) # ReLU
        self.logits = np.dot(self.a1, self.W2) + self.b2
        
        # 4. Softmax Probs
        exps = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        return self.probs

    def backward(self, x_ids, y_ids):
        B = x_ids.shape[0]
        # 1. Output Gradient
        dout = self.probs.copy()
        dout[np.arange(B), y_ids] -= 1.0
        dout /= B
        
        # 2. Weights W2, b2 Gradient
        dW2 = np.dot(self.a1.T, dout)
        db2 = np.sum(dout, axis=0, keepdims=True)
        
        # 3. Backprop through ReLU
        da1 = np.dot(dout, self.W2.T)
        da1[self.a1 <= 0] = 0
        
        # 4. Backprop through LayerNorm
        dbeta = np.sum(da1, axis=0, keepdims=True)
        dgamma = np.sum(da1 * self.z1_norm, axis=0, keepdims=True)
        dz1_norm = da1 * self.gamma
        dz1 = (dz1_norm - np.mean(dz1_norm, axis=1, keepdims=True) - self.z1_norm * np.mean(dz1_norm * self.z1_norm, axis=1, keepdims=True)) / self.std
        
        # 5. Backprop through W1, b1
        dW1 = np.dot(self.x_flat.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # 6. Backprop through Embeddings (Vectorized)
        dx_flat = np.dot(dz1, self.W1.T)
        dembs = dx_flat.reshape(B, self.context_size, self.emb_dim)
        dE = np.zeros_like(self.E)
        np.add.at(dE, x_ids, dembs) # This aggregates gradients for all indices in x_ids
        
        return [dE, dW1, db1, dgamma, dbeta, dW2, db2]

    def save(self, path):
        data = {f'p_{i}': p for i, p in enumerate(self.params)}
        data.update({f'm_{i}': m for i, m in enumerate(self.opt.ms)})
        data.update({f'v_{i}': v for i, v in enumerate(self.opt.vs)})
        data['t'] = self.opt.t
        np.savez(path, **data)
        print(f"💾 Checkpoint saved: {path}")

    def load(self, path):
        if not os.path.exists(path): return False
        try:
            with np.load(path, allow_pickle=True) as data:
                # Vocab safety check
                if data['p_0'].shape[0] != self.vocab_size:
                    print("⚠️ Vocab mismatch in checkpoint. Starting fresh.")
                    return False
                # Restore arrays in-place to maintain list references
                for i in range(len(self.params)):
                    self.params[i][:] = data[f'p_{i}']
                    self.opt.ms[i][:] = data[f'm_{i}']
                    self.opt.vs[i][:] = data[f'v_{i}']
                self.opt.t = int(data['t'])
                print(f"📂 Loaded successfully (Timestep: {self.opt.t})")
                return True
        except Exception as e:
            print(f"💥 Failed to load: {e}")
            if os.path.exists(path): os.remove(path)
            return False

# --- DATA & TRAINING ---
def get_clean_data(path):
    print(f"📖 Reading corpus from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        text = re.sub(r'[^a-z\s]', ' ', f.read().lower())
    words = text.split()
    counts = Counter(words)
    # Remove rare words
    words = [w for w in words if counts[w] > 1]
    vocab = sorted(list(set(words)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    print(f"✅ Vocab Size: {len(vocab)} | Total Tokens: {len(words)}")
    return [w2i[w] for w in words], vocab, w2i, i2w

def train_ultima():
    # Paths
    corpus_path = "/Users/indra/Desktop/DEEP_LEARNING/04_Neural_Framework/Architectures/nature_corpus.txt"
    ckpt_path = "/Users/indra/Desktop/DEEP_LEARNING/04_Neural_Framework/Architectures/nature_ultima.npz"
    
    # Setup
    data_ids, vocab, w2i, i2w = get_clean_data(corpus_path)
    model = Nature_SLM_Ultima(len(vocab))
    model.load(ckpt_path)
    
    data_np = np.array(data_ids)
    context_size = 5
    num_samples = len(data_ids) - context_size
    batch_size = 128
    
    print("🚀 TRAINING STARTED (Mac Mini M4 Optimized)")
    print("-" * 50)
    
    for epoch in range(100):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        epoch_loss = 0
        start_time = time.time()
        
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            
            # Efficient vectorized batch building
            b_x = np.array([data_np[j : j + context_size] for j in batch_idx])
            b_y = data_np[batch_idx + context_size]
            
            # Forward
            probs = model.forward(b_x)
            loss = -np.mean(np.log(probs[np.arange(len(b_y)), b_y] + 1e-15))
            
            if np.isnan(loss):
                print("💥 DIVERGENCE DETECTED. Resetting weights...")
                return
            
            epoch_loss += loss
            
            # Backward & Step
            grads = model.backward(b_x, b_y)
            model.opt.step(model.params, grads)
            
            if i % (batch_size * 500) == 0:
                print(f"Epoch {epoch} | {i/num_samples:.1%} | Loss: {loss:.4f}")
        
        # Save & Stats
        avg_loss = epoch_loss / (num_samples // batch_size)
        print(f"✅ Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s")
        model.save(ckpt_path)
        
        # --- GENERATION TEST ---
        sample_idx = np.random.randint(0, num_samples)
        ctx = data_np[sample_idx : sample_idx + context_size]
        prompt = " ".join([i2w[idx] for idx in ctx])
        gen = []
        temp_ctx = ctx.copy()
        
        for _ in range(20):
            p = model.forward(np.array([temp_ctx]))[0]
            # Temperature sampling
            p = np.exp(np.log(p + 1e-15) / 0.8)
            p /= p.sum()
            idx = np.random.choice(len(p), p=p)
            gen.append(i2w[idx])
            temp_ctx = np.append(temp_ctx[1:], idx)
        
        print(f"🔮 Prompt: \"{prompt}\"")
        print(f"   Result: \033[92m{' '.join(gen)}\033[0m")
        print("-" * 50)

if __name__ == "__main__":
    train_ultima()
