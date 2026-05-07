import numpy as np
from deep_slm_architect import ReLU
from slm_architect import Layer, Embedding, Flatten, Dense, Softmax, Sequential

# --- 📚 THE MINI-CORPUS ---
vocab = {"i": 0, "love": 1, "ai": 2, "is": 3, "deep": 4, "learning": 5}
inv_vocab = {v: k for k, v in vocab.items()}

# Training Pairs: (Input indices, Target one-hot)
X_train = np.array([
    [0, 1], # "i love"
    [2, 3], # "ai is"
    [4, 5], # "deep learning"
    [0, 1]  # "i love" (duplicate to show consistency)
])

# Targets: "ai" (2), "deep" (4), "is" (3), "deep" (4)
y_train = np.array([
    [0, 0, 1, 0, 0, 0], # ai
    [0, 0, 0, 0, 1, 0], # deep
    [0, 0, 0, 1, 0, 0], # is
    [0, 0, 0, 0, 1, 0]  # deep
])

def train_batched_slm():
    print("📦 STARTING BATCHED SLM TRAINING")
    print(f"Batch Size: {len(X_train)} sentences")
    print("=" * 60)

    # 🏗️ Build the Model
    # input: 2 words * 4 dims = 8
    model = Sequential([
        Embedding(vocab_size=6, dim=4),
        Flatten(),
        Dense(8, 16),
        ReLU(),
        Dense(16, 6),
        Softmax()
    ])

    # 🔄 Training Loop
    for epoch in range(1001):
        # 1. FORWARD (Batch pass)
        # Probs shape: [4 samples x 6 words]
        probs = model.forward(X_train)
        
        # 2. LOSS
        error = probs - y_train
        
        # 3. BACKWARD (Blame is averaged across batch)
        model.backward(error, learning_rate=0.05)
        
        if epoch % 200 == 0:
            avg_loss = np.mean(error**2)
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f}")

    print("\n🏁 TRAINING COMPLETE. TESTING KNOWLEDGE:")
    
    # Test on a batch
    test_sentences = ["i love", "ai is", "deep learning"]
    test_indices = np.array([
        [vocab[w] for w in s.split()] for s in test_sentences
    ])
    
    final_probs = model.forward(test_indices)
    
    for i, sentence in enumerate(test_sentences):
        pred_idx = np.argmax(final_probs[i])
        print(f"Input: '{sentence}' -> Predicted: '{inv_vocab[pred_idx]}'")

if __name__ == "__main__":
    train_batched_slm()
