import numpy as np

class SLM_Lab_V3:
    """
    SLM Lab V3: Context Windows & Sequence Memory
    The 'Thinking' layer now looks at a window of words instead of just one.
    """
    def __init__(self, vocab_size, emb_dim, context_size):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.context_size = context_size
        
        # 1. Identity Matrix (Embeddings)
        self.embeddings = np.random.randn(vocab_size, emb_dim) * 0.1
        
        # 2. Thinking Matrix (Weights)
        # Note: Input is now (emb_dim * context_size) because we concatenate the window
        input_dim = emb_dim * context_size
        self.W = np.random.randn(input_dim, vocab_size) * 0.1
        
        # Adam Memory
        self.m_emb, self.v_emb = np.zeros_like(self.embeddings), np.zeros_like(self.embeddings)
        self.m_W, self.v_W = np.zeros_like(self.W), np.zeros_like(self.W)
        self.t = 0

    def forward(self, batch_context_ids):
        """
        batch_context_ids: Shape (Batch, Context_Size)
        """
        batch_size = len(batch_context_ids)
        
        # 1. Identity Lookup: Fetch embeddings for each word in the window
        # embs shape: (Batch, Context_Size, Emb_Dim)
        self.embs = self.embeddings[batch_context_ids] 
        
        # 2. Concatenation: Flatten the window into a single 'Context Vector'
        # x shape: (Batch, Context_Size * Emb_Dim)
        self.x = self.embs.reshape(batch_size, -1) 
        
        # 3. Thinking: Project context vector to vocab logits
        self.logits = np.dot(self.x, self.W) # (Batch, Vocab)
        
        # 4. Softmax Choice
        exps = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        return self.probs

    def backward(self, batch_tar):
        batch_size = len(batch_tar)
        
        # 1. The Miracle Error (P - Y)
        dout = self.probs.copy()
        for i, target in enumerate(batch_tar):
            dout[i, target] -= 1.0
        dout /= batch_size # Normalize by batch size
        
        # 2. Gradients for Thinking Weights (dW)
        # dW = x.T * dout | Shape: (Context*Dim, Vocab)
        self.dW = np.dot(self.x.T, dout)
        
        # 3. Gradients for Identity (d_emb)
        # d_context = dout * W.T | Shape: (Batch, Context*Dim)
        d_context = np.dot(dout, self.W.T)
        
        # Reshape d_context back to (Batch, Context_Size, Emb_Dim)
        self.d_embs = d_context.reshape(batch_size, self.context_size, self.emb_dim)
        
        return dout

    def update(self, batch_context_ids, lr):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        # --- Update Thinking Weights (W) ---
        self.m_W = beta1 * self.m_W + (1 - beta1) * self.dW
        self.v_W = beta2 * self.v_W + (1 - beta2) * (self.dW**2)
        m_hat = self.m_W / (1 - beta1**self.t)
        v_hat = self.v_W / (1 - beta2**self.t)
        self.W -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # --- Update Identity Embeddings ---
        # Note: We must aggregate gradients if the same word appears multiple times in the batch/window
        for b in range(len(batch_context_ids)):
            for c in range(self.context_size):
                idx = batch_context_ids[b, c]
                grad = self.d_embs[b, c]
                
                self.m_emb[idx] = beta1 * self.m_emb[idx] + (1 - beta1) * grad
                self.v_emb[idx] = beta2 * self.v_emb[idx] + (1 - beta2) * (grad**2)
                
                m_h = self.m_emb[idx] / (1 - beta1**self.t)
                v_h = self.v_emb[idx] / (1 - beta2**self.t)
                self.embeddings[idx] -= lr * m_h / (np.sqrt(v_h) + eps)

def prepare_data(sentence, context_size):
    words = sentence.split()
    vocab = sorted(list(set(words)))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    
    data = []
    for i in range(len(words) - context_size):
        context = [word_to_id[words[j]] for j in range(i, i + context_size)]
        target = word_to_id[words[i + context_size]]
        data.append((context, target))
    
    return np.array(data, dtype=object), vocab, word_to_id, id_to_word

def slm_experiment_v3(epochs=2000, lr=0.01, batch_size=8, context_size=3):
    sentence = "the sun rises in the east and the moon shines in the west the sun sets in the west and the moon fades in the east"
    data, vocab, word_to_id, id_to_word = prepare_data(sentence, context_size)
    
    print(f"🔬 SLM LAB V3 | CONTEXT SIZE: {context_size} | VOCAB: {len(vocab)}")
    print("-" * 80)
    
    model = SLM_Lab_V3(len(vocab), 16, context_size)
    
    for epoch in range(epochs):
        np.random.shuffle(data)
        epoch_loss = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            b_ctx = np.array([p[0] for p in batch])
            b_tar = [p[1] for p in batch]
            
            probs = model.forward(b_ctx)
            loss = -np.mean(np.log(probs[np.arange(len(b_tar)), b_tar] + 1e-15))
            epoch_loss += loss
            
            model.backward(b_tar)
            model.update(b_ctx, lr)

            # --- DEEP CONTEXT TRACE (First batch, first epoch) ---
            if epoch == 0 and i == 0:
                print(f"\n🧠 THE CONTEXT X-RAY: HOW {model.context_size} WORDS BECOME ONE THOUGHT")
                print("-" * 80)
                print(f"1. WINDOW: { [id_to_word[idx] for idx in b_ctx[0]] }")
                print(f"2. INDIVIDUAL IDENTITIES: {model.embs[0].shape} ({model.context_size} words x {model.emb_dim} dim)")
                print(f"3. CONCATENATED CONTEXT: {model.x[0].shape} (1 vector of {model.context_size * model.emb_dim} dim)")
                print(f"4. PREDICTION: Thinking weights W {model.W.shape} mapped {model.context_size * model.emb_dim}D context to {len(vocab)}D probabilities.")
                
                print(f"\n⚡ THE GRADIENT REVERSE-CONCATENATION:")
                print(f"   Error ($dout$) hit W.T and produced a {model.context_size * model.emb_dim}D gradient.")
                print(f"   We sliced it back into {model.context_size} pieces of {model.emb_dim}D to update each word in the window.")
                print("-" * 80 + "\n")

        if epoch % 500 == 0:
            print(f"Epoch {epoch:4} | Loss: {epoch_loss / (len(data)/batch_size):.4f}")

    # --- TESTING ---
    print("\n🏁 CONTEXTUAL PREDICTIONS:")
    test_sentences = [
        "the sun",
        "the moon",
        "sun rises",
        "moon shines"
    ]
    
    for ts in test_sentences:
        words_in = ts.split()
        if len(words_in) != context_size:
            # Adjust test sentence to match context size
            full_words = "the sun rises in the east".split()
            words_in = full_words[:context_size]
            ts = " ".join(words_in)

        ctx_ids = [word_to_id[w] for w in words_in]
        probs = model.forward(np.array([ctx_ids]))
        pred_id = np.argmax(probs)
        print(f"Context: '{ts:15}' -> Next: '{id_to_word[pred_id]}'")

def mini_math_demo():
    """
    🔬 MINI MATH X-RAY: B=2, C=2, D=3, V=4
    Shows the raw numbers for every matrix operation.
    """
    print("\n" + "█" * 80)
    print(" 🔢 MINI MATH X-RAY: SEEING THE TENSORS COLLIDE (B=2, C=2, D=3, V=4)")
    print("█" * 80)

    # 1. SETUP
    vocab_size, emb_dim, context_size = 4, 16, 2
    model = SLM_Lab_V3(vocab_size, emb_dim, context_size)
    
    # 2. INPUT: Batch of 2 contexts (IDs)
    # Batch 0: Word 0 and Word 1
    # Batch 1: Word 2 and Word 3
    batch_ids = np.array([[0, 1], [2, 3]])
    targets = [1, 0] # Targets for the two batches
    
    print(f"\n1. INPUT IDs (B=2, C=2):\n{batch_ids}")
    
    # 3. FORWARD
    probs = model.forward(batch_ids)
    
    print(f"\n2. EMBEDDINGS (B=2, C=2, D=3):\n{model.embs}")
    print(f"\n3. CONCATENATED CONTEXT X (B=2, C*D=6):\n{model.x}")
    print(f"\n4. THINKING WEIGHTS W (C*D=6, V=4):\n{model.W[:3, :]} ... (showing half)")
    print(f"\n5. LOGITS (B=2, V=4):\n{model.logits}")
    print(f"\n6. PROBABILITIES (Softmax applied):\n{probs}")

    # 4. BACKWARD
    dout = model.backward(targets)
    
    print(f"\n7. THE ERROR BATON dout (P - Y):\n{dout}")
    print(f"\n8. WEIGHT GRADIENT dW (X.T * dout):\n{model.dW[:3, :]} ...")
    print(f"\n9. EMBEDDING GRADIENT d_embs (The Sliced Gradient):\n{model.d_embs}")
    
    print("\n" + "█" * 80)
    print(" ✅ MATH TRACE COMPLETE")
    print("█" * 80 + "\n")

if __name__ == "__main__":
    # RUN THE EXPERIMENT
    # slm_experiment_v3(epochs=3001, lr=0.01, context_size=2)
    
    # RUN THE MINI MATH DEMO
    mini_math_demo()
