import numpy as np

# --- 1. THE BLUEPRINT (Base Layer) ---
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError

# --- 2. THE LEGO BLOCKS ---

class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.weights = np.random.randn(vocab_size, dim) * 0.1

    def forward(self, input_indices):
        self.input = input_indices.astype(int)
        # Select rows from dictionary
        return self.weights[self.input]

    def backward(self, output_error, learning_rate):
        # output_error shape: [batch, seq, dim]
        # We only update the words that 'spoke'
        d_weights = np.zeros_like(self.weights)
        
        # Flattening sequence to indices
        indices = self.input.flatten()
        errors = output_error.reshape(-1, output_error.shape[-1])
        
        for i, idx in enumerate(indices):
            d_weights[idx] += errors[i]
            
        self.weights -= learning_rate * d_weights
        return None

class Flatten(Layer):
    def forward(self, input_data):
        self.input_shape = input_data.shape
        # [Batch, Seq, Dim] -> [Batch, Seq * Dim]
        return input_data.reshape(input_data.shape[0], -1)

    def backward(self, output_error, learning_rate):
        return output_error.reshape(self.input_shape)

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        return input_error

class Softmax(Layer):
    def forward(self, input_data):
        exp = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        self.output = exp / np.sum(exp, axis=-1, keepdims=True)
        return self.output

    def backward(self, output_error, learning_rate):
        # For simplicity with Cross-Entropy, we assume the error is already (pred - target)
        return output_error

# --- 3. THE MASTER CONTAINER (Sequential) ---

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, error, learning_rate):
        # Travel BACKWARD through the layers list
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

# --- 4. THE SLM TRAINING ---

def main():
    # Vocabulary
    vocab = {"i": 0, "love": 1, "ai": 2, "deep": 3, "learning": 4}
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # Architecture: [2 words] -> [Embed (4D)] -> [Flatten (8D)] -> [Dense (5 words)]
    model = Sequential([
        Embedding(vocab_size=5, dim=4),
        Flatten(),
        Dense(input_size=8, output_size=5),
        Softmax()
    ])

    # Data: "i love" -> "ai"
    X = np.array([[0, 1]]) # indices
    y = np.array([[0, 0, 1, 0, 0]]) # one-hot for "ai"

    print("🚀 TRAINING SEQUENTIAL SLM...")
    for epoch in range(501):
        # 1. FORWARD (One call!)
        probs = model.forward(X)
        
        # 2. LOSS (Error)
        error = probs - y
        
        # 3. BACKWARD (One call!)
        model.backward(error, learning_rate=0.1)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Pred: '{inv_vocab[np.argmax(probs)]}' | Prob: {np.max(probs):.4f}")

    print("\n🏁 ARCHITECTURE COMPLETE.")
    print("The model now handles the hand-offs between layers automatically.")

if __name__ == "__main__":
    main()
