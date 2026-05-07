import numpy as np

def the_symmetry_trap():
    print("👥 THE SYMMETRY TRAP: WHY WE NEED RANDOMNESS")
    print("=" * 60)

    # --- SCENARIO 1: THE CLONES (Identical Initialization) ---
    # Two neurons with the SAME weights
    w1 = 0.5
    w2 = 0.5
    
    x = 1.0
    target = 0.8
    lr = 0.1

    print("--- 🔴 SCENARIO 1: IDENTICAL START ---")
    print(f"Initial: Neuron1={w1}, Neuron2={w2}")
    
    # Forward
    pred1 = x * w1; pred2 = x * w2
    # Backward
    grad1 = (pred1 - target) * x
    grad2 = (pred2 - target) * x
    
    # Update
    w1 -= lr * grad1; w2 -= lr * grad2
    
    print(f"After Update: Neuron1={w1:.4f}, Neuron2={w2:.4f}")
    print("Result: They are still IDENTICAL. They are acting like a single neuron.")

    # --- SCENARIO 2: THE EXPERTS (Random Initialization) ---
    # Two neurons with SLIGHTLY DIFFERENT weights
    w3 = 0.51
    w4 = 0.49

    print("\n--- 🟢 SCENARIO 2: RANDOM START ---")
    print(f"Initial: Neuron3={w3}, Neuron4={w4}")
    
    # Forward
    pred3 = x * w3; pred4 = x * w4
    # Backward
    grad3 = (pred3 - target) * x
    grad4 = (pred4 - target) * x
    
    # Update
    w3 -= lr * grad3; w4 -= lr * grad4
    
    print(f"After Update: Neuron3={w3:.4f}, Neuron4={w4:.4f}")
    print("Result: They have moved to DIFFERENT values! They are becoming 'Experts'.")

    print("\n" + "=" * 60)
    print("💡 THE DEEP TRUTH:")
    print("If you start as clones, you die as clones. The math cannot 'break' the symmetry.")
    print("Randomness is the 'spark' that allows neurons to find their own unique jobs.")

if __name__ == "__main__":
    the_symmetry_trap()
