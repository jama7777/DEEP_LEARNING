import numpy as np

class SLM_Lab_V2:
    def __init__(self, vocab_size, emb_dim):
        self.embeddings = np.random.randn(vocab_size, emb_dim) * 0.1
        self.W = np.random.randn(emb_dim, vocab_size) * 0.1
        
        # Adam Memory
        self.m_emb, self.v_emb = np.zeros_like(self.embeddings), np.zeros_like(self.embeddings)
        self.m_W, self.v_W = np.zeros_like(self.W), np.zeros_like(self.W)
        self.t = 0

    def forward(self, batch_ids):
        # 1. Identity Lookup
        self.x = self.embeddings[batch_ids] # (Batch, Dim)
        # 2. Thinking
        self.logits = np.dot(self.x, self.W) # (Batch, Vocab)
        # 3. Softmax Choice
        exps = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        return self.probs

    def backward(self, batch_tar):
        batch_size = len(batch_tar)
        # 1. The Miracle Error (P - Y)
        dout = self.probs.copy()
        for i, target in enumerate(batch_tar):
            dout[i, target] -= 1.0
        dout /= batch_size
        
        # 2. Gradients
        self.dW = np.dot(self.x.T, dout)
        self.d_emb = np.dot(dout, self.W.T)
        return dout

    def update(self, batch_ids, lr, use_adam):
        self.t += 1
        if not use_adam:
            # Simple SGD
            self.W -= lr * self.dW
            for i, idx in enumerate(batch_ids):
                self.embeddings[idx] -= lr * self.d_emb[i]
        else:
            # Adam Optimization Logic
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # Update W
            self.m_W = beta1 * self.m_W + (1 - beta1) * self.dW
            self.v_W = beta2 * self.v_W + (1 - beta2) * (self.dW**2)
            m_hat = self.m_W / (1 - beta1**self.t)
            v_hat = self.v_W / (1 - beta2**self.t)
            self.W -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # Update Embeddings
            for i, idx in enumerate(batch_ids):
                self.m_emb[idx] = beta1 * self.m_emb[idx] + (1 - beta1) * self.d_emb[i]
                self.v_emb[idx] = beta2 * self.v_emb[idx] + (1 - beta2) * (self.d_emb[i]**2)
                m_h = self.m_emb[idx] / (1 - beta1**self.t)
                v_h = self.v_emb[idx] / (1 - beta2**self.t)
                self.embeddings[idx] -= lr * m_h / (np.sqrt(v_h) + eps)

def slm_experiment_lab(epochs=500, lr=0.01, batch_size=4, use_adam=True, use_shuffle=True):
    print(f"🔬 SLM LAB V2 | EPOCHS: {epochs} | LR: {lr} | BATCH: {batch_size} | ADAM: {use_adam} | SHUFFLE: {use_shuffle}")
    print("=" * 85)

    # 1. DATA: 20 Unique Words
    sentence = "the sun rises in the the sun rises in the sun in the sun rises in the sun east and the moon shines in the dark night while the world sleeps in peace"
    words = sentence.split()
    vocab = sorted(list(set(words)))
    vocab_size = len(vocab)
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}

    pairs = []
    for i in range(len(words) - 1):
        pairs.append((word_to_id[words[i]], word_to_id[words[i+1]]))

    # 2. INIT MODEL (Dimension 16)
    model = SLM_Lab_V2(vocab_size, 16)

    # 3. TRAINING
    for epoch in range(epochs):
        if use_shuffle: np.random.shuffle(pairs)
        
        epoch_loss = 0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            b_in = [p[0] for p in batch]
            b_tar = [p[1] for p in batch]
            
            probs = model.forward(b_in)
            loss = -np.mean(np.log(probs[np.arange(len(b_in)), b_tar] + 1e-15))
            epoch_loss += loss
            
            dout = model.backward(b_tar)
            model.update(b_in, lr, use_adam)

            # --- DEEP BACKPROP TRACE (First batch, first epoch only) ---
            if epoch == 0 and i == 0:
                print("\n🔥 THE DEEP BACKPROP TRACE: WHERE DOES THE ERROR GO?")
                print("-" * 85)
                print(f"1. FORWARD: Input Matrix X {model.x.shape} meet Weights W {model.W.shape}")
                print(f"   Resulting Logits Y {model.logits.shape}")
                
                print(f"\n2. CHOICE: Softmax turned Logits into Probabilities.")
                print(f"   Top Probability in Batch: {np.max(probs):.4f}")

                # Calculate dout (The Baton)
                dout = model.backward(b_tar)
                
                print(f"\n3. ERROR (The Baton): Shape {dout.shape}")
                print(f"   This 'Baton' now splits into TWO directions:")
                
                print(f"\n   A. TO THE DENSE WEIGHTS (dW):")
                print(f"      dW = X.T * Error")
                print(f"      This tells the 'Thinking' weights how to change.")
                print(f"      dW Sample (3x3):\n{model.dW[:3, :3]}")

                print(f"\n   B. TO THE WORD IDENTITY (d_emb):")
                print(f"      d_emb = Error * W.T")
                print(f"      This tells the 'Word' where to move in 16D space.")
                print(f"      d_emb Sample (Row 0):\n{model.d_emb[0, :5]} ...")

                print(f"\n4. FINAL DESTINATION: The Optimizer.")
                print("   The gradients (dW, d_emb) are sent to Adam.")
                print("   Adam uses 'Momentum' to decide the final step size.")
                print("-" * 85 + "\n")

        if epoch % 100 == 0:
            print(f"Epoch {epoch:3} | Avg Loss: {epoch_loss / (len(pairs)/batch_size):.4f}")

    # 4. RESULTS
    print("\n🏁 SAMPLE PREDICTIONS:")
    for test_word in ["the", "sun", "moon", "world", "dark","night","peace","the","sun","moon","world","dark","night","peace","the","sun","moon","world","dark","night","peace"]:
        idx = word_to_id[test_word]
        probs = model.forward([idx])
        pred_id = np.argmax(probs)
        print(f"Input: '{test_word:7}' -> Predicted Next: '{id_to_word[pred_id]:7}'")

if __name__ == "__main__":
    # CHANGE THESE PARAMETERS TO EXPERIMENT!
    slm_experiment_lab(
        epochs=10001, 
        lr= 0.01, 
        batch_size=10, 
        use_adam=True, 
        use_shuffle=True
    )
