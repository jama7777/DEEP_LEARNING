import numpy as np

def main():
    # 1. THE DATA (INPUTS)
    # Features: [Camera (1-10), Battery (1-10), Storage (1-10)]
    phone_names = ["SuperCam Phone", "Battery King", "Storage Beast", "Budget Balanced"]
    
    # We put them in a Matrix (each row is a phone)
    phones_data = np.array([
        [10, 4, 6],  # SuperCam: Great camera, weak battery
        [3, 10, 5],  # Battery King: Weak camera, amazing battery
        [5, 5, 10],  # Storage Beast: Amazing storage
        [6, 6, 6]    # Budget Balanced: Average at everything
    ])
    
    # 2. YOUR WEIGHTS (THE OPINION)
    # Let's say you care MOST about Camera, then Battery, then Storage.
    # We define weights that add up to 1.0 (or anything really)
    weights = np.array([0.7, 0.2, 0.1]) 
    
    print("--- User Preferences ---")
    print(f"Weight for Camera:  {weights[0]}")
    print(f"Weight for Battery: {weights[1]}")
    print(f"Weight for Storage: {weights[2]}\n")

    # 3. THE DOT PRODUCT (THE SCORE)
    # This multiplies every phone by your weights and gives a single score.
    scores = np.dot(phones_data, weights)
    
    # 4. THE BIAS (THE THRESHOLD)
    # Let's say you won't buy ANY phone unless its score is at least 7.0.
    # We represent this as a Bias of -7.0
    bias = -5.0
    
    print("--- Results (Threshold = 7.0) ---")
    for name, score in zip(phone_names, scores):
        final_decision = score + bias
        decision_str = "✅ BUY" if final_decision > 0 else "❌ REJECT"
        print(f"{name}: Score = {score:.2f} | Final = {final_decision:+.2f} | {decision_str}")
    
    # 5. PICK THE ABSOLUTE BEST
    winner_idx = np.argmax(scores)
    if scores[winner_idx] + bias > 0:
        print(f"\n🏆 The AI recommends buying: {phone_names[winner_idx]}")
    else:
        print(f"\n🚫 The AI says: None of these phones meet your standards (Bias: {bias})")

if __name__ == "__main__":
    main()
