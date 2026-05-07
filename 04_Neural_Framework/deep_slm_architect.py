import numpy as np
from slm_architect import Layer, Embedding, Flatten, Dense, Softmax, Sequential

# --- ⚡ THE NON-LINEAR SWITCH: ReLU ---
class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        # Filter: Keep positive, kill negative
        return np.maximum(0, input_data)

    def backward(self, output_error, learning_rate):
        # Only pass blame back to neurons that were 'ON' ( > 0)
        # If the neuron was 0, it was 'dead', so no blame flows back.
        input_error = output_error.copy()
        input_error[self.input <= 0] = 0
        return input_error

# --- 🚀 THE DEEP SLM ARCHITECT ---

def main():
    # Vocabulary
    vocab = {"i": 0, "love": 1, "ai": 2, "is": 3, "deep": 4}
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # 🏗️ Build a DEEPER Model
    # We add a hidden layer (16 neurons) and a ReLU switch
    model = Sequential([
        Embedding(vocab_size=5, dim=4),
        Flatten(),
        Dense(input_size=8, output_size=16), # Hidden Thinking Layer
        ReLU(),                              # The Filter
        Dense(input_size=16, output_size=5), # Final Decision Layer
        Softmax()
    ])

    # Data: "i love" -> "ai"
    X = np.array([[0, 1]])
    y = np.array([[0, 0, 1, 0, 0]])

    print("🚀 TRAINING THE DEEP SLM (with ReLU)...")
    print("Architecture: Embedding -> Dense -> ReLU -> Dense -> Softmax")
    
    for epoch in range(501):
        # Forward
        probs = model.forward(X)
        
        # Loss (Error)
        error = probs - y
        
        # Backward
        model.backward(error, learning_rate=0.1)
        
        if epoch % 100 == 0:
            loss = np.mean(error**2)
            print(f"Epoch {epoch} | Loss: {loss:.6f} | Pred: '{inv_vocab[np.argmax(probs)]}'")

    print("\n🏁 DEEP TRAINING COMPLETE.")
    print("By adding ReLU, we've allowed the network to have 'conditional' logic.")
    print("The first layer now finds 'features', and the second layer decides how to use them.")

if __name__ == "__main__":
    main()
