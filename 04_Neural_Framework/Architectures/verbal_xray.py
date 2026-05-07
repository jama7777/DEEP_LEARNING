import numpy as np
from xray_utils import show_detailed_math

# --- THE VERBAL "X-RAY" COMPONENTS ---
class EmbeddingXRay:
    def __init__(self, vocab_size, embedding_dim):
        self.weights = np.array([
            [ 0.1,  0.2, -0.1,  0.0], # Index 0: "i"
            [-0.2,  0.1,  0.3, -0.1], # Index 1: "love"
            [ 0.0,  0.0,  0.1,  0.1], # Index 2: "ai"
            [ 0.1, -0.1,  0.0,  0.2], # Index 3: "is"
            [-0.1,  0.2,  0.1,  0.0]  # Index 4: "deep"
        ])

    def forward(self, input_indices):
        # We look at the words one by one to keep the math 2D
        idx_1 = input_indices[0, 0]
        idx_2 = input_indices[0, 1]
        
        vec_1 = self.weights[idx_1].reshape(1, -1)
        vec_2 = self.weights[idx_2].reshape(1, -1)
        
        print("\n--- 🔎 STEP 1: THE DICTIONARY LOOKUP ---")
        show_detailed_math("Vector for 'i' (Index 0)", [self.weights[0:1]], vec_1, label="lookup")
        show_detailed_math("Vector for 'love' (Index 1)", [self.weights[1:2]], vec_2, label="lookup")
        
        return vec_1, vec_2

class DenseXRay:
    def __init__(self, input_size, output_size):
        # Specific weights to make the winner clear
        self.weights = np.zeros((input_size, output_size))
        self.weights[0, 2] = 1.0 # Pushing towards 'ai'
        self.weights[4, 2] = 1.0 
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.biases
        print("\n--- 🧠 STEP 2: THE COMBINATION (Dense Layer) ---")
        show_detailed_math("Z = Combined_Input @ Weights", [self.input, self.weights], self.z, operation="*")
        return self.z

def main():
    # Data: "i love"
    X = np.array([[0, 1]])
    
    emb = EmbeddingXRay(5, 4)
    dense = DenseXRay(8, 3) # Input: 8 (two 4D vectors), Output: 3 words

    print("🚀 STARTING THE VERBAL X-RAY (Step-by-Step Math)...")
    
    # 1. Lookup
    v1, v2 = emb.forward(X)
    
    # 2. Glue (Concatenate)
    flattened = np.concatenate([v1, v2], axis=1)
    print("\n--- 🧲 STEP 1.5: THE CONCATENATION ---")
    show_detailed_math("Combined Meaning [v1, v2]", [v1, v2], flattened, operation="+") # Using + as visual glue
    print("Logic: We have glued the 'i' vector and the 'love' vector into one 8D signal.")

    # 3. Dense
    z_out = dense.forward(flattened)
    
    # 4. Results
    words = ["i", "love", "ai"]
    print("\n--- 🎯 FINAL SCORES ---")
    for i, score in enumerate(z_out[0]):
        print(f"Word '{words[i]}': {score:.3f}")
    
    print(f"\n🏆 PREDICTION: '{words[np.argmax(z_out)]}'")

if __name__ == "__main__":
    main()
