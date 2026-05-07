import numpy as np

def adam_cancellation_xray():
    print("🥊 THE ADAM TUG-OF-WAR: POSITIVE vs. NEGATIVE")
    print("=" * 70)

    m = 0.0
    beta1 = 0.9
    
    # SCENARIO: 
    # Step 1: Tell the model to move RIGHT (+1)
    # Step 2: Tell the model to move LEFT  (-1)
    # Step 3: Tell the model to move LEFT  (-1)
    gradients = [1.0, 1.0, 1.0,1,1,1]

    print(f"{'Step':<5} | {'Grad':<6} | {'m (Bucket)':<12} | {'m_hat (Boosted)':<15} | {'Logic'}")
    print("-" * 70)

    for i, grad in enumerate(gradients, 1):
        # 1. Update the bucket (The Tug of War)
        m = beta1 * m + (1 - beta1) * grad
        
        # 2. Apply Boost
        m_hat = m / (1 - beta1**i)
        
        direction = "Right ➡️" if m_hat > 0 else "Left ⬅️"
        if abs(m_hat) < 0.1: direction = "CENTER 🛑"
        
        print(f"{i:<5} | {grad:<6.1f} | {m:<12.4f} | {m_hat:<15.4f} | {direction}")

    print("\n" + "=" * 70)
    print("💡 THE DEEP TRUTH:")
    print("1. Look at Step 2: Today was -1.0, but the bucket is almost ZERO (-0.05).")
    print("   Why? Because yesterday's +1.0 'cancelled' today's -1.0.")
    print("2. By Step 3: The model has finally 'forgotten' the Right turn")
    print("   and is now moving Left (-0.38).")
    print("\n[CONCLUSION]: beta1 handles the 'Weight of Time',")
    print("while the +/- Sign handles the 'Conflict of Direction'.")

if __name__ == "__main__":
    adam_cancellation_xray()
