import numpy as np

def batch_symmetry_logic():
    print("🪞 BATCH vs. SYMMETRY: THE MIRROR TEST")
    print("=" * 60)

    # 1. ONE BRAIN (The Weight)
    weight = 0.5
    
    # 2. DIFFERENT PEOPLE (The Batch)
    input_A = 10.0 # Tall Person
    input_B = 1.0  # Short Person
    input_C = -5.0 # Person upside down

    print(f"The Mirror (Weight) is: {weight}")

    # 3. THE REFLECTIONS (Forward Pass)
    # Even though the weight is the SAME, the results are DIFFERENT
    # because the INPUTS are different.
    output_A = input_A * weight
    output_B = input_B * weight
    output_C = input_C * weight

    print("\n--- ⚡ THE REFLECTIONS ---")
    print(f"Input A (10.0) -> Output: {output_A}")
    print(f"Input B ( 1.0) -> Output: {output_B}")
    print(f"Input C (-5.0) -> Output: {output_C}")

    print("\n" + "=" * 60)
    print("💡 THE CLARITY:")
    print("1. Is there symmetry? NO. Each output is unique.")
    print("2. Why? Because Symmetry is a problem of NEURONS, not DATA.")
    print("\n[POINT]: You only get a 'Symmetry Problem' if you have")
    print("two different weights that start with the same value.")
    print("Batching is just using one weight to look at many different faces.")

if __name__ == "__main__":
    batch_symmetry_logic()
