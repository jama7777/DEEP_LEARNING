import numpy as np
from slm_architect import Embedding, Flatten, Dense, Softmax, Sequential, Layer

# --- 🚀 THE PROFESSIONAL PIECES ---

class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)
    def backward(self, output_error, learning_rate):
        input_error = output_error.copy()
        input_error[self.input <= 0] = 0
        return input_error

class AdamOptimizer:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = {} # Momentum memory
        self.v = {} # Variance memory
        self.t = 0  # Time step

    def update(self, layer_id, weights, grads):
        self.t += 1
        if layer_id not in self.m:
            self.m[layer_id] = np.zeros_like(weights)
            self.v[layer_id] = np.zeros_like(weights)
        
        # 1. Momentum (Average of gradients)
        self.m[layer_id] = self.beta1 * self.m[layer_id] + (1 - self.beta1) * grads
        # 2. Scaling (Average of variance)
        self.v[layer_id] = self.beta2 * self.v[layer_id] + (1 - self.beta2) * (grads**2)
        
        # Bias correction
        m_hat = self.m[layer_id] / (1 - self.beta1**self.t)
        v_hat = self.v[layer_id] / (1 - self.beta2**self.t)
        
        return weights - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

# --- 📚 THE TRAINING DATA ---

vocab = {"i": 0, "love": 1, "ai": 2, "is": 3, "deep": 4, "learning": 5}
inv_vocab = {v: k for k, v in vocab.items()}

# Pairs: (Input indices, Target word index)
data = [
    ([0, 1], 2), # i love -> ai
    ([2, 3], 4), # ai is -> deep
    ([4, 5], 3), # deep learning -> is
    ([0, 1], 4), # i love -> deep (Alternate)
]

X_train = np.array([item[0] for item in data])
y_train = np.zeros((len(data), 6))
for i, item in enumerate(data):
    y_train[i, item[1]] = 1.0

# --- 🏗️ THE MASTER SESSION ---

def main():
    print("🎓 LINGUISTIC MASTER TRAINER")
    print("=" * 60)

    # 1. Build Model
    model = Sequential([
        Embedding(vocab_size=6, dim=4),
        Flatten(),
        Dense(8, 16),
        ReLU(),
        Dense(16, 6),
        Softmax()
    ])

    opt = AdamOptimizer(lr=0.02)

    print("🚀 Training on Mini-Corpus...")
    for epoch in range(10010):
        # Forward
        probs = model.forward(X_train)
        
        # Cross-Entropy Gradient: (Pred - Target)
        error = probs - y_train
        
        # Backward (Customized to use Adam)
        # Note: We are manually applying Adam to each layer's weights
        grad = error
        for i in reversed(range(len(model.layers))):
            layer = model.layers[i]
            if isinstance(layer, Dense):
                # Calculate Gradients
                w_grad = np.dot(layer.input.T, grad)
                b_grad = np.sum(grad, axis=0, keepdims=True)
                
                # Update using ADAM instead of simple subtraction
                layer.weights = opt.update(f"L{i}_W", layer.weights, w_grad)
                layer.biases = opt.update(f"L{i}_B", layer.biases, b_grad)
                
                # Pass error back
                grad = np.dot(grad, layer.weights.T)
            elif isinstance(layer, Embedding):
                # Update words that spoke
                indices = layer.input.flatten()
                errs = grad.reshape(-1, grad.shape[-1])
                for j, idx in enumerate(indices):
                    layer.weights[idx] = opt.update(f"Emb_{idx}", layer.weights[idx], errs[j])
            else:
                grad = layer.backward(grad, 0.0) # ReLU/Flatten don't use LR

        if epoch % 500 == 0:
            loss = -np.mean(np.sum(y_train * np.log(probs + 1e-8), axis=1))
            # Track the 'i love' samples (Index 0 and 3)
            p_ai = probs[0, 2]   # Prob of 'ai' for first 'i love'
            p_deep = probs[0, 4] # Prob of 'deep' for first 'i love'
            print(f"Epoch {epoch:5} | Loss: {loss:.4f} | 'i love' split: [ai: {p_ai*100:4.1f}% | deep: {p_deep*100:4.1f}%]")
            if p_ai > 0.99 or p_deep > 0.99:
                print("⚠️  COLLAPSE DETECTED: The model has picked a side and stopped being ambiguous!")

    print("\n🏁 TRAINING COMPLETE. MODEL KNOWLEDGE:")
    final_probs = model.forward(X_train)
    for i, item in enumerate(data):
        sentence = " ".join([inv_vocab[idx] for idx in item[0]])
        pred_idx = np.argmax(final_probs[i])
        print(f"Input: '{sentence}' -> Predicted: '{inv_vocab[pred_idx]}' ({final_probs[i, pred_idx]*100:.1f}%)")

if __name__ == "__main__":
    main()
