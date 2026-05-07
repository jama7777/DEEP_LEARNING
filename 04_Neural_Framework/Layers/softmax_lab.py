import numpy as np

def negative_microscope(value):
    print(f"\n--- 🔬 THE NEGATIVE MICROSCOPE: Input Score = {value} ---")
    
    # The Power of e
    result = np.exp(value)
    
    print(f"Action: e^{value}")
    if value < 0:
        print(f"Deep Point: Even though {value} is negative, the result is POSITIVE: {result:.10f}")
        print(f"Effect: This word is being 'SILENCED'. It will contribute almost 0 to the sum.")
    elif value == 0:
        print(f"Deep Point: e^0 is always 1.0. This is the 'Neutral' starting point.")
    else:
        print(f"Deep Point: e^{value} is an EXPLOSION: {result:.2f}")
        print(f"Effect: This word is SHOUTING for attention!")

def main():
    print("🚀 SOFTMAX: THE NEGATIVE HANDLER")
    print("="*60)

    # 1. Neutral (0)
    negative_microscope(0)

    # 2. Strong Disagreement (-5)
    negative_microscope(-5)

    # 3. Total Silence (-50)
    negative_microscope(-50)

    # --- THE SQUISH PROOF ---
    scores = [10, 0, -50]
    exps = np.exp(scores)
    probs = exps / np.sum(exps)
    
    print("\n--- 🏁 THE FINAL SQUISH PROOF ---")
    print(f"Scores: {scores}")
    print(f"Exps:   {exps}")
    print(f"Percentages: {probs * 100} %")
    print("\nLogic: The -50 became so small that it effectively disappeared from the 100% total.")

if __name__ == "__main__":
    main()
