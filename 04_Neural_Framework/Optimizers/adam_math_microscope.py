import numpy as np

def adam_math_microscope():
    print("🧬 THE ADAM MATH: ECHOES and BOOSTS")
    print("=" * 70)

    m = 0.0
    grad = 1.0
    beta1 = 0.9

    print(f"{'Step':<5} | {'Raw m':<10} | {'Boost Factor':<15} | {'Final m_hat':<12} | {'Logic'}")
    print("-" * 70)

    for i in range(1, 1000):
        # 1. The Raw Echo
        m = beta1 * m + (1 - beta1) * grad
        
        # 2. The Boost Factor (1 - beta^i)
        # This factor starts small (0.1) and grows to 1.0
        boost_denominator = (1 - beta1**i)
        
        # 3. The Final Boosted Momentum
        m_hat = m / boost_denominator
        
        logic = "Full Power" if i == 1 else "Stabilizing"
        print(f"{i:<5} | {m:<10.4f} | {boost_denominator:<15.4f} | {m_hat:<12.4f} | {logic}")

    print("\n" + "=" * 70)
    print("💡 THE DEEP TRUTH:")
    print("1. Look at Step 1: Raw 'm' was only 0.1, but we BOOSTED it to 1.0.")
    print("2. Why? Because the 'Boost Factor' realized we only have 10% of history,")
    print("   so it multiplied the result by 10 to make it 'Full Strength'.")
    print("3. By Step 5, 'm' is growing naturally, so the 'Boost' is getting smaller.")
    print("\n[CONCLUSION]: Adam uses these echos to ensure that even at Step 1,")
    print("the model is moving at the correct speed.")

if __name__ == "__main__":
    adam_math_microscope()
