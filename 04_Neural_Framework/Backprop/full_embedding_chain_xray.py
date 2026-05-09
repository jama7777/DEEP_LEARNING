import numpy as np

class FullVerbalChain:
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        # 1. THE EMBEDDING (The Identity)
        self.embeddings = np.random.randn(vocab_size, emb_dim) * 0.1
        
        # 2. THE DENSE LAYER (The Thinking)
        self.W_dense = np.random.randn(emb_dim, hidden_dim) * 0.1
        
        self.cache = {}

    def forward(self, word_id):
        # Step A: Lookup Embedding
        emb_vector = self.embeddings[word_id].reshape(1, -1)
        
        # Step B: Multiply by Dense Weights (The Thinking)
        # Prediction = Embedding * Weights
        prediction = np.dot(emb_vector, self.W_dense)
        
        self.cache = {'id': word_id, 'emb': emb_vector}
        return prediction

    def backward(self, dout):
        word_id = self.cache['id']
        emb_vector = self.cache['emb']
        
        # --- THE DEEP MATH ---
        
        # 1. Calculate Gradient for Dense Weights
        # dW = Emb.T * dout
        dW_dense = np.dot(emb_vector.T, dout)
        
        # 2. THE TRANSLATION: Calculate Gradient for the Embedding itself!
        # This is how the 'Incoming Error' reaches the word vector.
        # dEmb = dout * W_dense.T
        d_emb_vector = np.dot(dout, self.W_dense.T)
        
        # 3. UPDATE THE EMBEDDING TABLE
        # We only update the row for that specific word
        self.embeddings[word_id] -= 0.1 * d_emb_vector.flatten()
        
        # 4. UPDATE THE DENSE WEIGHTS
        self.W_dense -= 0.1 * dW_dense
        
        return d_emb_vector

def deep_chain_xray():
    print("⛓️ THE FULL VERBAL CHAIN X-RAY: FROM LOSS TO EMBEDDING")
    print("=" * 75)

    # 1. SETUP (Word 'Sun' -> 4D Embedding -> 1D Score)
    chain = FullVerbalChain(vocab_size=10, emb_dim=4, hidden_dim=1)
    sun_id = 1
    
    print(f"INITIAL EMBEDDING for 'Sun':\n{chain.embeddings[sun_id]}")
    
    # 2. FORWARD PASS
    pred = chain.forward(sun_id)
    print(f"\nFORWARD: Prediction (Thinking) = {pred[0,0]:.4f}")

    # 3. BACKWARD PASS (The CEO says 'Too High by 1.0!')
    dout = np.array([[1.0]]) 
    d_emb = chain.backward(dout)
    
    print(f"\nBACKWARD: The Error translated through the Dense Weights:")
    print(f"Translated Error for Embedding (d_emb):\n{d_emb}")
    
    print(f"\nNEW EMBEDDING for 'Sun':\n{chain.embeddings[sun_id]}")

    print("\n" + "-" * 75)
    print("💡 THE SUPREME INNER MATH:")
    print("1. The word 'Sun' didn't see the 1.0 error directly.")
    print("2. It saw (1.0 * W_dense.T).")
    print("3. If the Dense Weight was positive, the Embedding moves one way.")
    print("4. If the Dense Weight was negative, the Embedding moves the OTHER way.")
    print("RESULT: The Embedding is 'Molded' by the layer above it!")

if __name__ == "__main__":
    deep_chain_xray()
