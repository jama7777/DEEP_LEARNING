import numpy as np

# --- 🧱 THE VERBAL LEGO: EMBEDDING LAYER ---
class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        # A dictionary of weights: [Vocab_Size x Dimensions]
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.input = None

    def forward(self, input_indices):
        self.input = input_indices.astype(int)
        # Select the 'Meaning Vectors' for these specific words
        # Input shape: [Batch, Sequence_Length]
        # Output shape: [Batch, Sequence_Length, Embedding_Dim]
        self.output = self.weights[self.input]
        return self.output

    def backward(self, output_error, learning_rate):
        # We only update the words that were actually used!
        d_weights = np.zeros_like(self.weights)
        
        # This is the 'Push': Move the word vector based on the error
        for i, idx in enumerate(self.input.flatten()):
            d_weights[idx] += output_error.reshape(-1, output_error.shape[-1])[i]
        
        self.weights -= learning_rate * d_weights
        return None # Embedding is the first layer, nothing to pass back to

# --- ⚡ DENSE LAYER (Re-using from our previous work) ---
class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        return input_error

class Softmax:
    def forward(self, input_data):
        exp = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        self.output = exp / np.sum(exp, axis=-1, keepdims=True)
        return self.output

    def backward(self, output_error, learning_rate):
        return output_error # Softmax backward is usually combined with Cross-Entropy

# --- 🎯 THE VERBAL DATASET ---
# Vocabulary: {i:0, love:1, ai:2, is:3, deep:4}
vocab = {"i":0, "love":1, "ai":2, "is":3, "deep":4}
inv_vocab = {v: k for k, v in vocab.items()}

# Training Data: ["i", "love"] -> "ai"
X = np.array([[0, 1]]) # indices for "i", "love"
y_true = np.array([[0, 0, 1, 0, 0]]) # One-hot for "ai" (index 2)

# 🏗️ Build the Verbal Stack
# 1. Embedding (Words to Vectors)
# 2. Flatten (Combine word meanings)
# 3. Dense (Logic)
# 4. Softmax (Probability)
emb = Embedding(5, 4) # 5 words, 4-dimensional meaning
dense = Dense(8, 5)   # 2 words * 4 dims = 8 inputs. 5 output categories.
softmax = Softmax()

print("🚀 TRAINING THE MINI-LANGUAGE MODEL...")
for epoch in range(1001):
    # Forward
    e_out = emb.forward(X)
    flattened = e_out.reshape(1, -1) # Flatten the two 4-dim vectors into one 8-dim vector
    d_out = dense.forward(flattened)
    probs = softmax.forward(d_out)
    
    # Loss (Simple MSE for now)
    error = probs - y_true
    
    # Backward
    d_err = dense.backward(error, 0.1)
    emb.backward(d_err.reshape(1, 2, 4), 0.1)

    if epoch % 200 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {epoch} | Loss: {loss:.6f} | Pred: {inv_vocab[np.argmax(probs)]}")

print("\n🏁 VERBAL TEST:")
test_input = "i love"
indices = np.array([[vocab[w] for w in test_input.split()]])
e_out = emb.forward(indices)
d_out = dense.forward(e_out.reshape(1, -1))
probs = softmax.forward(d_out)
predicted_word = inv_vocab[np.argmax(probs)]

print(f"Input: '{test_input}' -> Predicted Next Word: '{predicted_word}'")
