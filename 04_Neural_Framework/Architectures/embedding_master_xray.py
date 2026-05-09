import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, dim):
        # Every word starts as a random point in space
        self.weights = np.random.randn(vocab_size, dim) * 0.1
        self.cache = {}

    def forward(self, idx):
        # We just 'look up' the row for that word
        self.cache['idx'] = idx
        return self.weights[idx].reshape(1, -1)

    def backward(self, dout):
        idx = self.cache['idx']
        # The 'Incoming Error' tells us how to move this specific word
        # We only update the row for the word we just saw!
        dW = np.zeros_like(self.weights)
        dW[idx] = dout
        return dW

def semantic_drift_xray():
    print("🌌 THE SEMANTIC DRIFT X-RAY: WATCHING WORDS MOVE")
    print("=" * 70)

    # 1. SETUP
    vocab = ["the", "sun", "moon", "rises"]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    emb = EmbeddingLayer(len(vocab), 2) # 2D space so we can visualize
    
    def get_similarity(w1, w2):
        v1 = emb.weights[word_to_id[w1]]
        v2 = emb.weights[word_to_id[w2]]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    print(f"Initial Similarity (Sun vs Moon): {get_similarity('sun', 'moon'):.4f}")
    print("-" * 70)

    # 2. TRAINING (The Deep Part)
    # We simulate training where 'sun' and 'moon' both get 'up-voted' for the same task
    lr = 0.1
    for epoch in range(100):
        # Simulate the network saying 'sun' and 'moon' should both move towards a 'bright' feature
        # (Simplified backprop for the demo)
        target_direction = np.array([1.0, 1.0])
        
        # Update 'sun'
        emb.forward(word_to_id['sun'])
        dW_sun = emb.backward(target_direction)
        emb.weights += dW_sun * lr
        
        # Update 'moon'
        emb.forward(word_to_id['moon'])
        dW_moon = emb.backward(target_direction)
        emb.weights += dW_moon * lr

    # 3. THE RESULT
    print(f"Final Similarity (Sun vs Moon):   {get_similarity('sun', 'moon'):.4f}")
    
    print("\n💡 THE DEEP TRUTH:")
    print("1. We never TOLD the computer that 'sun' and 'moon' are similar.")
    print("2. But because they both received the same 'Incoming Error' (Target Direction),")
    print("   their vectors drifted towards the same point in space.")
    print("3. This is how AI understands meaning! It's just 'Mathematical Proximity'.")

if __name__ == "__main__":
    semantic_drift_xray()
