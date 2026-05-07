import numpy as np

def layernorm_deep_dive():
    print("⚖️ LAYERNORM DEEP DIVE: EQUALIZATION + FREEDOM")
    print("=" * 70)

    # 1. RAW DATA (A word embedding)
    x = np.array([2.0, -1.0, 0.5, 4.0])
    print(f"Original Data: {x}")

    # 2. THE EQUALIZATION (The 'Locked' Step)
    mean = np.mean(x)
    std  = np.std(x)
    x_norm = (x - mean) / (std + 1e-8)
    
    print(f"\nStep 1: Normalized (Mean 0, Std 1):")
    print(f"{x_norm}")
    print("The data is now stable, but it's 'trapped' near zero.")

    # 3. THE FREEDOM KNOBS (Gamma & Beta)
    # Let's say the model has learned that it needs this data to be 
    # twice as loud and shifted by +5.
    gamma = 2.0 # Scale
    beta  = 5.0 # Shift
    
    # Final Output = (Normalized * Gamma) + Beta
    output = (x_norm * gamma) + beta

    print(f"\nStep 2: Gamma ({gamma}) & Beta ({beta}) Applied:")
    print(f"{output}")

    print("\n" + "=" * 70)
    print("💡 THE DEEP TRUTH:")
    print("1. The Normalization (Step 1) provides STABILITY.")
    print("2. Gamma & Beta (Step 2) provide EXPRESSION.")
    print("3. By combining them, the model can 'choose' its own volume")
    print("   without ever worrying about the numbers exploding to infinity.")

if __name__ == "__main__":
    layernorm_deep_dive()
