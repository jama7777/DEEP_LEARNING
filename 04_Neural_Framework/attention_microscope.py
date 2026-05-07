import numpy as np

def attention_microscope():
    print("🔭 THE ATTENTION MICROSCOPE: HOW WORDS 'LOOK' AT EACH OTHER")
    print("=" * 65)

    # 1. THE DATA: "the bank of the river"
    # We'll zoom in on 'Bank' and 'River'
    # Each word has a 4D vector
    bank_vec  = np.array([0.1, 0.9, -0.1, 0.1]) # Strong 'Place' feature
    river_vec = np.array([0.0, 0.8, 0.4, -0.2]) # Strong 'Nature/Place' feature
    money_vec = np.array([0.9, 0.1, -0.5, 0.0]) # Strong 'Finance' feature

    print("--- 🧠 THE CONTEXT TEST ---")
    print(f"Word 'Bank' Vector:  {bank_vec}")
    print(f"Word 'River' Vector: {river_vec}")
    print(f"Word 'Money' Vector: {money_vec}")

    # 2. THE DOT PRODUCT MATCH (The 'Attention Score')
    # How much does 'Bank' like 'River' vs 'Money'?
    score_bank_river = np.dot(bank_vec, river_vec)
    score_bank_money = np.dot(bank_vec, money_vec)

    print("\n--- ⚡ THE MATCHMAKING (Dot Product) ---")
    print(f"Bank · River = {score_bank_river:.4f} (High Match!)")
    print(f"Bank · Money = {score_bank_money:.4f} (Low Match)")

    # 3. THE SOFTMAX (The Spotlight)
    # Turn scores into 0% to 100%
    scores = np.array([score_bank_river, score_bank_money])
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)

    print("\n--- 🔦 THE SPOTLIGHT (Softmax) ---")
    print(f"Attention on River: {probs[0]*100:.1f}%")
    print(f"Attention on Money: {probs[1]*100:.1f}%")
    print("Logic: Because the vectors 'match' more, the model shines its light on 'River'.")

    # 4. THE VALUE BLEND (The Result)
    # We take 87% of River's meaning and 13% of Money's meaning
    context_vector = (probs[0] * river_vec) + (probs[1] * money_vec)

    print("\n--- 💎 THE NEW MEANING (Contextualized Vector) ---")
    print(f"Original Bank:  {bank_vec}")
    print(f"Enhanced Bank:  {context_vector}")
    print("\n💡 DEEP POINT:")
    print("The word 'Bank' has now absorbed the 'Nature' context from 'River'.")
    print("It is no longer just a word; it is a word with a SURROUNDING.")

if __name__ == "__main__":
    attention_microscope()
