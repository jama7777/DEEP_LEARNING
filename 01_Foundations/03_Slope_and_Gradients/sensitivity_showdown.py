import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sensitivity_showdown():
    print("--- SENSITIVITY SHOWDOWN ---")
    print("Comparing different 'Learning Strategies'\n")
    
    # We will test scores from -5 (Strong NO) to +5 (Strong YES)
    scores = np.linspace(-5, 5, 11)
    
    print(f"{'Score':<6} | {'Pred (p)':<10} | {'S1 (p)':<10} | {'S2 (1-p)':<10} | {'S3 (p*(1-p))':<15}")
    print("-" * 65)
    
    for z in scores:
        p = sigmoid(z)
        s1 = p              # Strategy 1: Only care about confidence
        s2 = 1 - p          # Strategy 2: Only care about doubt
        s3 = p * (1 - p)    # Strategy 3: The real Sigmoid Derivative
        
        # Visualize the magnitude of S3 with stars
        stars = "*" * int(s3 * 40)
        
        print(f"{z:<6.1f} | {p:<10.3f} | {s1:<10.3f} | {s2:<10.3f} | {s3:<10.3f} {stars}")

    print("\nOBSERVATIONS:")
    print("1. S1 (p) keeps increasing. It would make the AI go crazy at high scores.")
    print("2. S2 (1-p) keeps decreasing. It would make the AI blind at high scores.")
    print("3. S3 (p*(1-p)) peaks in the middle (0.250) and dies at BOTH ends.")
    print("   This is the 'Golden Zone' where learning actually happens!")

if __name__ == "__main__":
    sensitivity_showdown()
