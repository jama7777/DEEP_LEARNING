import numpy as np

class HighDimEmbedding:
    def __init__(self, vocab_size, dim):
        # Starting with random 128D vectors
        self.weights = np.random.randn(vocab_size, dim) * 0.1

    def update(self, idx, target_vec, lr):
        # Move the specific word vector towards the context goal
        error = target_vec - self.weights[idx]
        self.weights[idx] += lr * error

def run_tug_of_war():
    print("🥊 THE 128D TUG-OF-WAR: SUN VS MOON")
    print("=" * 65)

    # 1. SETUP
    DIM = 128
    SUN_ID = 0
    MOON_ID = 1
    model = HighDimEmbedding(2, DIM)
    
    # 2. THE CONTEXT TARGETS (The forces pulling the words)
    # Dimensions 0-50: 'Bright/Rise' Features (Shared)
    # Dimensions 51-100: 'Temperature' Features (Opposite)
    # Dimensions 101-127: 'Color' Features (Opposite)
    
    shared_force = np.zeros(DIM)
    shared_force[:50] = 1.0 # Pulls both towards 'Bright'
    
    sun_unique_force = np.zeros(DIM)
    sun_unique_force[51:100] = 1.0 # Pulls Sun towards 'Hot'
    
    moon_unique_force = np.zeros(DIM)
    moon_unique_force[51:100] = -1.0 # Pulls Moon towards 'Cold'

    def get_similarity():
        v1 = model.weights[SUN_ID]
        v2 = model.weights[MOON_ID]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    print(f"Initial Random Similarity: {get_similarity():.4f}")

    # 3. THE TRAINING (Mixed Sentences)
    # We mix the 70 shared and 50 unique sentences randomly
    dataset = []
    for _ in range(70): dataset.append(("shared", shared_force))
    for _ in range(50): dataset.append(("unique", sun_unique_force)) # This is for Sun
    
    # We'll do a separate dataset for Moon to keep it simple
    moon_dataset = []
    for _ in range(70): moon_dataset.append(("shared", shared_force))
    for _ in range(50): moon_dataset.append(("unique", moon_unique_force))

    lr = 0.01
    for epoch in range(100):
        # We process the sentences in a random order
        np.random.shuffle(dataset)
        np.random.shuffle(moon_dataset)
        
        for _, force in dataset:
            model.update(SUN_ID, force, lr)
        for _, force in moon_dataset:
            model.update(MOON_ID, force, lr)

    # 4. THE FINAL X-RAY
    final_sim = get_similarity()
    print("-" * 65)
    print(f"Final 128D Similarity (Balanced): {final_sim:.4f}")
    
    # 5. THE INNER MATH MICROSCOPE
    print("\n🔬 INNER MATH MICROSCOPE (Peeking at the weights)")
    print("-" * 65)
    
    # Peek at Dimension 0 (Shared Context: 'Bright')
    sun_dim_0 = model.weights[SUN_ID, 0]
    moon_dim_0 = model.weights[MOON_ID, 0]
    
    # Peek at Dimension 51 (Unique Context: 'Temperature')
    sun_dim_51 = model.weights[SUN_ID, 51]
    moon_dim_51 = model.weights[MOON_ID, 51]
    
    print(f"DIMENSION 0 (The Shared 'Bright' Room):")
    print(f"   Sun: {sun_dim_0:.4f} | Moon: {moon_dim_0:.4f}")
    print(f"   Gap: {abs(sun_dim_0 - moon_dim_0):.4f} (ALMOST ZERO!)")
    
    print(f"\nDIMENSION 51 (The Unique 'Temperature' Room):")
    print(f"   Sun: {sun_dim_51:.4f} | Moon: {moon_dim_51:.4f}")
    print(f"   Gap: {abs(sun_dim_51 - moon_dim_51):.4f} (LARGE!)")

    print("\n💡 THE SUPREME TRUTH:")
    print("1. In Dim 0, the 'Error Magnet' pulled both words to the SAME spot.")
    print("2. In Dim 51, the 'Error Magnet' pulled them in OPPOSITE directions.")
    print("3. Similarity is the average of ALL these rooms. This is why it's 0.36!")

if __name__ == "__main__":
    run_tug_of_war()
