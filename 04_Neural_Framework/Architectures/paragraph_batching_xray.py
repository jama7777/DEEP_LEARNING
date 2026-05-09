import numpy as np

def paragraph_batching_demo():
    print("📦 THE PARAGRAPH BATCHER: FROM TEXT TO MATRICES")
    print("=" * 65)

    # 1. THE TEXT (Simulated 10-word paragraph)
    text = "the sun rises in the east and sets in the west"
    words = text.split()
    print(f"ORIGINAL TEXT: {text}")

    # 2. CREATE PAIRS (The Sliding Window)
    pairs = []
    for i in range(len(words) - 1):
        pairs.append((words[i], words[i+1]))
    
    print(f"\nTOTAL PAIRS CREATED: {len(pairs)}")
    print(f"FIRST 3 PAIRS: {pairs[:3]}")

    # 3. SHUFFLE (The Deck Shuffle)
    np.random.shuffle(pairs)
    print(f"\nSHUFFLED PAIRS: {pairs[:3]} ... (Random order!)")

    # 4. BATCHING (Batch Size = 3)
    batch_size = 3
    print(f"\nDIVIDING INTO BATCHES (Size {batch_size}):")
    
    for i in range(0, len(pairs), batch_size):
        current_batch = pairs[i:i+batch_size]
        
        # In a real model, these would become a MATRIX
        batch_inputs = [p[0] for p in current_batch]
        batch_targets = [p[1] for p in current_batch]
        
        print(f"  BATCH {i//batch_size + 1}:")
        print(f"    Inputs:  {batch_inputs}")
        print(f"    Targets: {batch_targets}")
        print(f"    ACTION:  One Weight Update for these {len(current_batch)} pairs.")

if __name__ == "__main__":
    paragraph_batching_demo()
