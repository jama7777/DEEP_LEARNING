import numpy as np

def adam_pizza_math():
    print("🍕 THE ADAM PIZZA MATH: SCALING SLICES TO WHOLE MEALS")
    print("=" * 70)

    beta = 0.9
    print(f"{'Step':<5} | {'Slices Collected':<18} | {'Pizza Fullness':<18} | {'Final Meal'}")
    print("-" * 70)

    m = 0.0
    grad = 1.0 # The 'Whole Pizza' target

    for i in range(1, 6):
        # Add a new slice (10% of today)
        m = beta * m + (1 - beta) * grad
        
        # Calculate how much of a 'full' pizza we expect to have by now
        fullness = (1 - beta**i)
        
        # Scale the partial slices to a 'Full Meal'
        meal = m / fullness
        
        print(f"{i:<5} | {m:<18.4f} | {fullness:<18.4f} | {meal:<10.4f}")

    print("\n" + "=" * 70)
    print("💡 THE DEEP TRUTH:")
    print("1. In Step 1, you only have 0.10 of a pizza.")
    print("2. But you know you only have 10% of the capacity.")
    print("3. By dividing, you realize that if you had the WHOLE thing, it would be 1.0.")
    print("\n[CONCLUSION]: Adam 'extrapolates' from your tiny history")
    print("to guess what the 'Real Trend' is from Step 1.")

if __name__ == "__main__":
    adam_pizza_math()
