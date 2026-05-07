import numpy as np

def batch_vs_params_xray():
    print("🧠 THE SHARED BRAIN: BATCH vs. CONNECTIVITY")
    print("=" * 60)

    # 1. THE BRAIN (Weights) - These are FIXED for the whole batch
    # Let's say 2 Inputs -> 3 Hidden Neurons
    # Total Parameters: 2 * 3 = 6 weights
    weights = np.array([
        [0.1, 0.2, 0.3], # Connections from Input A
        [0.4, 0.5, 0.6]  # Connections from Input B
    ])
    
    print("--- 🏗️ THE STRUCTURE (The Parameters) ---")
    print("These 6 weights are the ONLY thing the model 'remembers'.")
    print(weights)

    # 2. THE BATCH (Inputs) - Different data using the same brain
    # Batch Size: 4 different 'people' talking to the brain
    batch_inputs = np.array([
        [1.0, 0.0], # Person 1: Active Input A
        [0.0, 1.0], # Person 2: Active Input B
        [1.0, 1.0], # Person 3: Both Active
        [0.5, 0.5]  # Person 4: Half Active
    ])

    print("\n--- 📦 THE BATCH (The Data) ---")
    print("4 different inputs entering the same brain at once:")
    print(batch_inputs)

    # 3. THE CRUNCH (The Dot Product)
    # Result = Batch @ Weights
    # [4 x 2] @ [2 x 3] = [4 x 3]
    outputs = np.dot(batch_inputs, weights)

    print("\n--- ⚡ THE RESULT (Hidden Activations) ---")
    print("Each row is the result for a different person in the batch.")
    for i in range(len(batch_inputs)):
        print(f"Person {i+1} used the SAME weights but got result: {outputs[i]}")

    print("\n" + "=" * 60)
    print("💡 THE BIG REVEAL:")
    print(f"1. Total weights used: {weights.size} (The fixed brain)")
    print(f"2. Total inputs processed: {batch_inputs.size} (The batch of data)")
    print("\n[POINT]: We didn't need 4 sets of weights. We used ONE set of weights")
    print("to calculate 4 different futures at the exact same time.")
    print("This is why GPUs are so fast—they are great at this 'one brain, many inputs' math.")

if __name__ == "__main__":
    batch_vs_params_xray()
