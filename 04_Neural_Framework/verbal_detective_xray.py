import numpy as np

def verbal_detective_xray():
    print("🕵️ THE VERBAL DETECTIVE: TRACING BLAME TO THE DICTIONARY")
    print("=" * 65)

    # 1. THE CRIME SCENE (The error for 'ai')
    # Let's say we have 8D blame for the phrase 'i love'
    # [i1, i2, i3, i4,  L1, L2, L3, L4]
    dl_dcombined = np.array([-0.5, 0.2, 0.0, 0.1,  0.8, -0.4, 0.1, 0.0])
    
    print("--- 📢 STEP 1: THE 8D BLAME SIGNAL ---")
    print("The detective has found 8 points of blame for the phrase 'i love':")
    print(dl_dcombined)

    # 2. THE SPLIT (dL_vi and dL_vlove)
    dl_vi = dl_dcombined[:4]    # First 4 features (The 'i' part)
    dl_vlove = dl_dcombined[4:] # Last 4 features (The 'love' part)

    print("\n--- ✂️ STEP 2: THE BLAME SPLIT ---")
    print(f"Blame assigned to 'i':    {dl_vi}")
    print(f"Blame assigned to 'love': {dl_vlove}")
    print("Logic: Because 'i' was the first word, it takes the first 4 pieces of blame.")

    # 3. THE DICTIONARY (emb_weights)
    # 5 words, 4 dims
    emb_weights = np.zeros((5, 4))
    
    print("\n--- 📖 STEP 3: THE DICTIONARY UPDATE ---")
    print("Dictionary Row 0 (i) before:    ", emb_weights[0])
    print("Dictionary Row 1 (love) before: ", emb_weights[1])
    
    # APPLY UPDATE (Nudging the dictionary)
    lr = 0.5
    emb_weights[0] -= lr * dl_vi
    emb_weights[1] -= lr * dl_vlove
    
    print("\n--- ✅ AFTER UPDATE ---")
    print("Dictionary Row 0 (i) after:     ", emb_weights[0])
    print("Dictionary Row 1 (love) after:  ", emb_weights[1])
    print("Dictionary Row 2 (ai) after:    ", emb_weights[2], " (NO CHANGE!)")

    print("\n" + "=" * 65)
    print("💡 DEEP SUMMARY:")
    print("1. dL_dCombined is the 'Global Blame' for the whole sentence.")
    print("2. [:, :4] is the 'Slicer' that finds which word to blame.")
    print("3. idx_i is the 'Address' in the dictionary where we send the blame.")
    print("4. Only the words that SPOKE are changed. 'ai' (Row 2) is silent, so its vector is safe.")

if __name__ == "__main__":
    verbal_detective_xray()
