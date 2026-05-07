import numpy as np

def adam_bias_deep_zoom():
    print("🪑 THE ADAM BIAS: IGNORING THE EMPTY CHAIRS")
    print("=" * 70)

    beta1 = 0.9
    grad = 1.0 # Every person walking in is '1.0'
    m = 0.0

    print(f"{'Step':<5} | {'How Full is the Room?':<25} | {'Raw m (The Lie)':<15} | {'m_hat (The Truth)':<15}")
    print("-" * 70)

    for i in range(1, 11):
        # 1. Update Raw m (Includes the empty chairs from 'm=0')
        m = beta1 * m + (1 - beta1) * grad
        
        # 2. Calculate Room Fullness (The Bias Correction factor)
        room_fullness = (1 - beta1**i)
        
        # 3. Get the Truth
        m_hat = m / room_fullness
        
        print(f"{i:<5} | {room_fullness*100:<24.1f}% | {m:<15.4f} | {m_hat:<15.4f}")

    print("\n" + "=" * 70)
    print("💡 THE DEEP TRUTH:")
    print("1. In Step 1, the 'Raw m' thinks the average is 0.1 because it counts the zeros.")
    print("2. But 'Room Fullness' knows the room is only 10% occupied.")
    print("3. $0.1 / 0.1 = 1.0$. The Truth is restored!")
    print("\n[CONCLUSION]: We never do this in normal gradients because they don't")
    print("have 'Memory'. But because Adam remembers the past (which starts at 0),")
    print("it MUST correct for those starting zeros.")

if __name__ == "__main__":
    adam_bias_deep_zoom()
