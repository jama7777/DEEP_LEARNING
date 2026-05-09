import numpy as np
import os
import time

print("🚀 RUNNING VERSION: V2.1 (VERBAL SANITATION ACTIVE)")

class Nature_SLM_Pro:
    """
    Nature SLM Pro: Deep MLP Architecture with LayerNorm & Checkpointing
    Optimized for training on the Mac Mini M4.
    """
    def __init__(self, vocab_size, emb_dim=128, context_size=5, hidden_dim=256):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.context_size = context_size
        self.hidden_dim = hidden_dim
        
        # 1. PARAMETERS
        # Identity
        self.E = np.random.randn(vocab_size, emb_dim) * 0.01
        # Thinking Layer 1
        self.W1 = np.random.randn(context_size * emb_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        # LayerNorm Parameters
        self.gamma = np.ones((1, hidden_dim))
        self.beta = np.zeros((1, hidden_dim))
        # Output Layer
        self.W2 = np.random.randn(hidden_dim, vocab_size) * 0.01
        self.b2 = np.zeros((1, vocab_size))
        
        # 2. ADAM MEMORY
        self.params = [self.E, self.W1, self.b1, self.gamma, self.beta, self.W2, self.b2]
        self.ms = [np.zeros_like(p) for p in self.params]
        self.vs = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, x_ids):
        # x_ids: (Batch, Context_Size)
        batch_size = x_ids.shape[0]
        
        # 1. Embed & Concat
        self.embs = self.E[x_ids] # (B, CS, D)
        self.x_flat = self.embs.reshape(batch_size, -1) # (B, CS*D)
        
        # 2. Hidden Layer (Pre-activation)
        self.z1 = np.dot(self.x_flat, self.W1) + self.b1 # (B, HD)
        
        # 3. Layer Normalization
        mu = np.mean(self.z1, axis=1, keepdims=True)
        var = np.var(self.z1, axis=1, keepdims=True)
        self.z1_norm = (self.z1 - mu) / np.sqrt(var + 1e-8)
        self.z1_ln = self.gamma * self.z1_norm + self.beta
        
        # 4. Activation (ReLU)
        self.a1 = np.maximum(0, self.z1_ln) # (B, HD)
        
        # 5. Output Logits
        self.logits = np.dot(self.a1, self.W2) + self.b2 # (B, VS)
        
        # 6. Softmax
        exps = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        return self.probs

    def backward(self, x_ids, y_ids):
        batch_size = x_ids.shape[0]
        
        # 1. Output Error (dout)
        dout = self.probs.copy()
        dout[np.arange(batch_size), y_ids] -= 1.0
        dout /= batch_size
        
        # 2. Back to W2, b2
        dW2 = np.dot(self.a1.T, dout)
        db2 = np.sum(dout, axis=0, keepdims=True)
        
        # 3. Back to a1 (ReLU)
        da1 = np.dot(dout, self.W2.T)
        da1[self.a1 <= 0] = 0 # ReLU derivative
        
        # 4. Back to LayerNorm
        dbeta = np.sum(da1, axis=0, keepdims=True)
        dgamma = np.sum(da1 * self.z1_norm, axis=0, keepdims=True)
        
        # Back through LN math (simplified LN grad)
        dz1_norm = da1 * self.gamma
        dz1 = (1.0/self.hidden_dim) * (self.hidden_dim * dz1_norm - np.sum(dz1_norm, axis=1, keepdims=True) - self.z1_norm * np.sum(dz1_norm * self.z1_norm, axis=1, keepdims=True))
        
        # 5. Back to W1, b1
        dW1 = np.dot(self.x_flat.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # 6. Back to Embeddings
        dx_flat = np.dot(dz1, self.W1.T)
        dembs = dx_flat.reshape(batch_size, self.context_size, self.emb_dim)
        
        # Gradient for E (Sparse update later)
        self.grads = [None, dW1, db1, dgamma, dbeta, dW2, db2]
        self.dembs = dembs
        return self.grads

    def update(self, x_ids, lr):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        # Update dense params
        for i in range(1, len(self.params)):
            grad = self.grads[i]
            self.ms[i] = beta1 * self.ms[i] + (1 - beta1) * grad
            self.vs[i] = beta2 * self.vs[i] + (1 - beta2) * (grad**2)
            m_hat = self.ms[i] / (1 - beta1**self.t)
            v_hat = self.vs[i] / (1 - beta2**self.t)
            self.params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Update Embeddings (Sparse)
        # Note: We aggregate gradients for the same index in the batch
        for b in range(x_ids.shape[0]):
            for c in range(self.context_size):
                idx = x_ids[b, c]
                g = self.dembs[b, c]
                self.ms[0][idx] = beta1 * self.ms[0][idx] + (1 - beta1) * g
                self.vs[0][idx] = beta2 * self.vs[0][idx] + (1 - beta2) * (g**2)
                m_h = self.ms[0][idx] / (1 - beta1**self.t)
                v_h = self.vs[0][idx] / (1 - beta2**self.t)
                self.E[idx] -= lr * m_h / (np.sqrt(v_h) + eps)

    def save(self, path):
        np.savez(path, E=self.E, W1=self.W1, b1=self.b1, gamma=self.gamma, beta=self.beta, W2=self.W2, b2=self.b2, t=self.t)
        print(f"💾 Checkpoint saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            data = np.load(path)
            # Shape Safety Check
            if data['E'].shape[0] != self.vocab_size:
                print(f"⚠️ Warning: Checkpoint vocab size ({data['E'].shape[0]}) does not match current vocab size ({self.vocab_size}).")
                print("   Skipping load to prevent crashes. Training from scratch.")
                return False
                
            self.E = data['E']
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.gamma = data['gamma']
            self.beta = data['beta']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.t = int(data['t'])
            # Reset params list to point to new arrays
            self.params = [self.E, self.W1, self.b1, self.gamma, self.beta, self.W2, self.b2]
            print(f"📂 Checkpoint loaded from {path} (Timestep: {self.t})")
            return True
        return False

import re
from collections import Counter

def get_data():
    path = "/Users/indra/Desktop/DEEP_LEARNING/04_Neural_Framework/Architectures/nature_corpus.txt"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    
    # 1. VERBAL SANITATION (Regex Cleaning)
    # Remove everything except letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 2. TOKENIZATION
    words = text.split()
    
    # 3. FREQUENCY FILTERING (Optional but high-impact)
    # Removing "hapax legomena" (words appearing once) to cut noise
    counts = Counter(words)
    words = [w for w in words if counts[w] > 1]
    
    vocab = sorted(list(set(words)))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    
    print(f"🧹 Sanitation Complete: Vocab reduced from ~29k to {len(vocab)}")
    return [word_to_id[w] for w in words], vocab, word_to_id, id_to_word

def train_nature_slm():
    # 1. LOAD DATA
    data_ids, vocab, w2i, i2w = get_data()
    vocab_size = len(vocab)
    context_size = 5
    batch_size = 128
    lr = 0.001
    
    print(f"🌲 NATURE SLM PRO | VOCAB: {vocab_size} | DATA: {len(data_ids)} words")
    print("-" * 60)
    
    # 2. INIT MODEL
    model = Nature_SLM_Pro(vocab_size, emb_dim=128, context_size=context_size, hidden_dim=256)
    checkpoint_path = "/Users/indra/Desktop/DEEP_LEARNING/04_Neural_Framework/Architectures/nature_slm_checkpoint.npz"
    model.load(checkpoint_path)
    
    # 3. PREPARE WINDOWS (Memory Efficient Slicing)
    num_samples = len(data_ids) - context_size
    # We use a trick to avoid giant list appending:
    data_np = np.array(data_ids)
    
    # 4. TRAINING LOOP
    epochs = 100
    for epoch in range(epochs):
        # Shuffle indices at the start of each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        epoch_loss = 0
        start_time = time.time()
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Efficiently build the batch
            # b_x shape: (batch_size, context_size)
            b_x = np.array([data_np[idx : idx + context_size] for idx in batch_indices])
            b_y = data_np[batch_indices + context_size]
            
            probs = model.forward(b_x)
            loss = -np.mean(np.log(probs[np.arange(len(b_y)), b_y] + 1e-15))
            epoch_loss += loss
            
            model.backward(b_x, b_y)
            model.update(b_x, lr)
            
            if i % (batch_size * 500) == 0:
                print(f"Batch {i}/{num_samples} | Current Loss: {loss:.4f}")

        # Save checkpoint after each epoch
        num_batches = num_samples // batch_size
        avg_loss = epoch_loss / num_batches
        print(f"✅ Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")
        model.save(checkpoint_path)
        
        # 5. LIVE GENERATION SAMPLE (See how the model is "thinking")
        sample_idx = np.random.randint(0, num_samples)
        context_ids = list(data_np[sample_idx : sample_idx + context_size])
        original_context = " ".join([i2w[idx] for idx in context_ids])
        
        generated = []
        temp_ids = context_ids[:]
        
        for _ in range(20):
            window = np.array([temp_ids[-context_size:]])
            probs = model.forward(window)[0]
            
            # Sampling with Temperature (0.7 for balance)
            probs = np.log(probs + 1e-15) / 0.7
            exp_probs = np.exp(probs - np.max(probs))
            probs = exp_probs / np.sum(exp_probs)
            next_id = np.random.choice(len(probs), p=probs)
            
            generated.append(i2w[next_id])
            temp_ids.append(next_id)
            
        print(f"🔮 EPOCH {epoch} GENERATION SAMPLE:")
        print(f"   Prompt: \"{original_context}\"")
        print(f"   Result: \033[92m{' '.join(generated)}\033[0m")
        print("-" * 60)

if __name__ == "__main__":
    train_nature_slm()
