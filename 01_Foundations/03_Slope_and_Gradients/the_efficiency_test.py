import numpy as np
import time

def efficiency_test():
    target_value = 42.42
    
    print("--- THE EFFICIENCY BATTLE ---")
    print(f"Target Value to Find: {target_value}\n")

    # --- METHOD 1: RANDOM GUESSING (Brute Force) ---
    print("AI 1: Random Guessing (The Blind Way)")
    start_time = time.time()
    guesses = 0
    best_guess = 0
    while True:
        guesses += 1
        guess = np.random.uniform(-100, 100)
        if abs(guess - target_value) < 0.01:
            best_guess = guess
            break
        if guesses > 100000: # Give up after 100k
            break
    
    print(f"Found: {best_guess:.2f} | Tries: {guesses} | Time: {time.time() - start_time:.4f}s")
    print("-" * 50)

    # --- METHOD 2: GRADIENT DESCENT (The Slope Way) ---
    print("AI 2: Gradient Descent (The Smart Way)")
    start_time = time.time()
    weight = 0.0
    lr = 0.1
    steps = 0
    while True:
        steps += 1
        # Calculate Slope
        slope = 2 * (weight - target_value)
        # Update
        weight -= lr * slope
        
        if abs(weight - target_value) < 0.01:
            break
            
    print(f"Found: {weight:.2f} | Steps: {steps} | Time: {time.time() - start_time:.4f}s")

    print("\nDEEP ANALYSIS:")
    print(f"Gradient Descent was {guesses / steps:.0f} times more efficient than Random Guessing!")
    print("If we had 1000 variables, Random Guessing would take TRILLIONS of years.")
    print("Gradient Descent would still take less than 1 second.")

if __name__ == "__main__":
    efficiency_test()
