import numpy as np

def layer_norm_equalizer():
    print("⚖️ THE LAYERNORM EQUALIZER: CALMING THE DATA")
    print("=" * 65)

    # 1. THE MESSY DATA (Exploding and Vanishing)
    # This represents a vector that has become unstable
    messy_vector = np.array([100.5, -50.2, 0.001, 10.5])
    
    print(f"Messy Vector: {messy_vector}")
    print(f"Max: {np.max(messy_vector):.2f} | Min: {np.min(messy_vector):.2f}")
    print("This vector is too 'loud'! It will break the next layer.")

    # 2. THE EQUALIZATION MATH
    # Step A: Find the Average (The Center)
    mean = np.mean(messy_vector)
    
    # Step B: Find the Standard Deviation (The Scale)
    std = np.std(messy_vector)
    
    # Step C: The Equalization
    # We subtract the mean and divide by std
    equalized = (messy_vector - mean) / (std + 1e-8)

    print("\n" + "-" * 65)
    print(f"Mean: {mean:.2f} | Std: {std:.2f}")
    print("-" * 65)

    # 3. THE CALM DATA
    print(f"Equalized Vector: {equalized}")
    print(f"New Mean: {np.mean(equalized):.1f} (Centered at Zero)")
    print(f"New Std:  {np.std(equalized):.1f} (Scaled to 1.0)")

    print("\n" + "=" * 65)
    print("💡 THE DEEP TRUTH:")
    print("1. LayerNorm ensures that no matter how 'crazy' the math gets,")
    print("   the numbers entering the next layer are always 'Standard'.")
    print("2. This is why you can stack 100 layers in a Transformer without it")
    print("   exploding into NaNs (Not a Number).")

if __name__ == "__main__":
    layer_norm_equalizer()
