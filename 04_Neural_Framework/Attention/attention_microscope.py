import numpy as np

def attention_microscope():
    print("🧠 THE ATTENTION MICROSCOPE: HOW WORDS FIND EACH OTHER")
    print("=" * 70)

    # 1. OUR SENTENCE: "The river bank"
    # Simplified embeddings for 'river' and 'bank'
    river = np.array([1.0, 0.0, 0.5]) # Water-related
    bank  = np.array([0.8, 0.0, 0.4]) # Can be water or money
    cat   = np.array([0.0, 1.0, 0.0]) # Unrelated
    
    # 2. THE DOT PRODUCT (The Similarity Check)
    # This is how words 'Score' their relationship
    score_river_bank = np.dot(river, bank)
    score_river_cat  = np.dot(river, cat)
    
    print(f"Similarity (River vs. Bank): {score_river_bank:.2f} (High Match! 🤝)")
    print(f"Similarity (River vs. Cat):  {score_river_cat:.2f}  (No Match! ❌)")

    # 3. THE SOFTMAX (The Voting)
    # We turn scores into 'Attention Weights' (percentages)
    scores = np.array([score_river_bank, score_river_cat])
    weights = np.exp(scores) / np.sum(np.exp(scores))
    
    print("\n" + "-" * 70)
    print(f"Attention Weight (Bank): {weights[0]*100:.1f}%")
    print(f"Attention Weight (Cat):  {weights[1]*100:.1f}%")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("💡 THE DEEP TRUTH:")
    print("1. In a Transformer, every word 'scans' every other word.")
    print("2. The 'River' word 'Attention-ed' the word 'Bank' because their")
    print("   vectors pointed in the same direction.")
    print("3. This allows the model to 'mix' the definition of River into Bank,")
    print("   so it knows we are talking about nature, not finance.")

if __name__ == "__main__":
    attention_microscope()
