import numpy as np

def normalization_battleground():
    print("🥊 THE NORMALIZATION BATTLEGROUND: BATCH VS LAYER")
    print("=" * 70)

    # Imagine a batch of 3 samples, each with 4 features (e.g., word embeddings)
    # Shape: (Batch_Size, Features) -> (3, 4)
    batch_data = np.array([
        [10.0, 20.0, 30.0, 40.0],  # Sample 1 (Loud)
        [ 1.0,  2.0,  3.0,  4.0],  # Sample 2 (Quiet)
        [-5.0,  0.0,  5.0, 10.0]   # Sample 3 (Centered)
    ])

    print("RAW BATCH DATA (3 Samples, 4 Features each):")
    print(batch_data)
    print("-" * 70)

    # --- BATCH NORM SELECTION ---
    # Logic: "I want to compare this sample to other samples in the batch."
    # Used in: CNNs, Vision.
    # Problem: If Batch Size = 1, it breaks!
    bn_mean = np.mean(batch_data, axis=0) # Mean across the BATCH (column-wise)
    bn_std  = np.std(batch_data, axis=0)
    batch_normed = (batch_data - bn_mean) / (bn_std + 1e-8)

    print("\n📦 BATCH NORM (Normalizing across the BATCH):")
    print(f"Mean per feature: {bn_mean}")
    print(batch_normed)
    print("-> Use this for: Computer Vision (CNNs).")
    print("-> Risk: Small batch sizes make it 'shaky'.")

    # --- LAYER NORM SELECTION ---
    # Logic: "I want to normalize this sample based ONLY on its own features."
    # Used in: Transformers, RNNs, NLP.
    # Benefit: Works perfectly even with Batch Size = 1.
    ln_mean = np.mean(batch_data, axis=1, keepdims=True) # Mean across FEATURES (row-wise)
    ln_std  = np.std(batch_data, axis=1, keepdims=True)
    layer_normed = (batch_data - ln_mean) / (ln_std + 1e-8)

    print("\n🍰 LAYER NORM (Normalizing across the LAYER/FEATURES):")
    print(f"Mean per sample:\n{ln_mean}")
    print(layer_normed)
    print("-> Use this for: Transformers (NLP), RNNs.")
    print("-> Strength: Ultra-stable regardless of batch size.")

    print("\n" + "=" * 70)
    print("📈 HOW WE SCALE: GAMMA & BETA")
    print("-" * 70)
    
    # Let's take one LayerNormed sample and 'Scale' it
    sample = layer_normed[0] # The first sample
    gamma = 0.5  # Squish the volume (Scale)
    beta  = 10.0 # Move to a new baseline (Shift)
    
    scaled_output = (sample * gamma) + beta
    
    print(f"LayerNormed Sample: {sample}")
    print(f"Final Scaled Output (Gamma={gamma}, Beta={beta}):")
    print(f"{scaled_output}")
    print("\n💡 SELECTION RULE OF THUMB:")
    print("1. Working with sequences/text? -> Use LayerNorm.")
    print("2. Working with images/large batches? -> Use BatchNorm.")
    print("3. Batch size is tiny (1 or 2)? -> Avoid BatchNorm, use LayerNorm/GroupNorm.")

if __name__ == "__main__":
    normalization_battleground()
