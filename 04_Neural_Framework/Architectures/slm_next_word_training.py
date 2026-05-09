import numpy as np
from master_stable_block import StableVerbalBlock

class EmbeddingLayer:
    def __init__(self, vocab_size, dim):
        self.weights = np.random.randn(vocab_size, dim) * 0.1
        self.cache = {}

    def forward(self, idx):
        self.cache['idx'] = idx
        return self.weights[idx].reshape(1, -1)

    def backward(self, d_vector):
        idx = self.cache['idx']
        dW = np.zeros_like(self.weights)
        dW[idx] = d_vector
        # Update weights (Simple gradient descent)
        self.weights -= 0.1 * dW

class SLM_NextWord:
    def __init__(self, vocab_size, emb_dim):
        self.emb = EmbeddingLayer(vocab_size, emb_dim)
        self.block = StableVerbalBlock(emb_dim, vocab_size) # Predicts vocab scores

    def forward(self, input_id):
        # 1. Identity
        x = self.emb.forward(input_id)
        # 2. Thinking
        logits = self.block.forward(x)
        # 3. Choice (Softmax)
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        return probs

    def train_step(self, input_id, target_id):
        # A. FORWARD
        probs = self.forward(input_id)
        
        # B. CALCULATE ERROR (The Miracle: P - Y)
        dout = probs.copy()
        dout[0, target_id] -= 1.0 # This is the entire calculus!
        
        # C. BACKWARD (The Baton Pass)
        # 1. Through the Block
        d_emb = self.block.backward(dout)
        # 2. To the Embedding
        self.emb.backward(d_emb)
        
        # Calculate Loss for tracking
        loss = -np.log(probs[0, target_id] + 1e-15)
        return loss

def sequence_training_lab():
    print("🎬 THE NEXT-WORD SEQUENTIAL LAB: TEACHING THE AI TO SPEAK")
    print("=" * 65)

    # 1. DATA
    sentence = "the sun rises in the east"
    words = sentence.split()
    vocab = sorted(list(set(words)))
    vocab_size = len(vocab)
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}

    # Create Input-Target Pairs
    pairs = []
    for i in range(len(words) - 1):
        pairs.append((word_to_id[words[i]], word_to_id[words[i+1]]))

    # 2. INITIALIZE SLM
    # We must make sure Input Dim == Output Dim for the Residual Connection!
    # So we use vocab_size for both.
    model = SLM_NextWord(vocab_size, vocab_size)

    # 3. TRAINING LOOP
    epochs = 5000
    for epoch in range(epochs):
        total_loss = 0
        for inp, tar in pairs:
            loss = model.train_step(inp, tar)
            total_loss += loss
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:3} | Loss: {total_loss:.4f}")

    # 4. FINAL TEST
    print("\n🏁 FINAL SEQUENCE TEST:")
    for inp, tar in pairs:
        probs = model.forward(inp)
        pred_id = np.argmax(probs)
        pred_word = id_to_word[pred_id]
        truth_word = id_to_word[tar]
        status = "✅" if pred_word == truth_word else "❌"
        print(f"Input: '{id_to_word[inp]:5}' | Target: '{truth_word:5}' | Pred: '{pred_word:5}' {status}")

    print("\n" + "-" * 65)
    print("💡 THE DEEP VICTORY:")
    print("1. We taught the model the RELATION between words.")
    print("2. 'The' now points to 'Sun' in mathematical space.")
    print("3. 'Sun' now points to 'Rises'.")
    print("4. This is exactly how LLMs generate text—one word at a time!")

if __name__ == "__main__":
    sequence_training_lab()
