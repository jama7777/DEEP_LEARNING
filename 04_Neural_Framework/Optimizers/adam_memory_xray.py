import numpy as np

def adam_memory_xray():
    print("🏚️ THE ADAM MEMORY GRAVEYARD: HOW OLD GRADIENTS FADE")
    print("=" * 70)

    beta1 = 0.9
    step = 5
    
    print(f"ANALYSIS OF STEP {step}:")
    print("-" * 40)
    
    total_percent = 0
    for i in range(step):
        # The math: (1 - beta) * beta^(age)
        # Age 0 is today, Age 1 is yesterday, etc.
        power = (1 - beta1) * (beta1 ** i)
        total_percent += power
        
        day = "Today" if i == 0 else f"{i} days ago"
        print(f"  {day:<12}: {power*100:>5.1f}% of the original signal remains")

    print("-" * 40)
    print(f"💰 TOTAL RAW POWER: {total_percent*100:.1f}%")
    print(f"🚀 BOOST FACTOR:    1 / {total_percent:.4f} = {1/total_percent:.2f}x")
    
    print("\n" + "=" * 70)
    print("💡 THE DEEP TRUTH:")
    print("1. In Step 5, the model only has 40.9% of a 'full' history.")
    print("2. So, m_hat = m / 0.409. It multiplies the bucket by 2.44x.")
    print("3. This makes the 40.9% feel like 100% power.")
    print("\n[CONCLUSION]: Memory is just a sum of 'Faded Echoes'.")
    print("Bias correction is the 'Volume Knob' that keeps the echoes loud.")

if __name__ == "__main__":
    adam_memory_xray()
