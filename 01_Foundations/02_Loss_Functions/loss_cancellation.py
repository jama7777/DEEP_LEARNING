def test_without_squaring():
    print("--- SCENARIO: AI misses twice ---")
    error1 = -10 # Missed left
    error2 = 10  # Missed right
    
    # WRONG WAY: Just adding
    wrong_total = error1 + error2
    
    # RIGHT WAY: Squaring (MSE)
    correct_total = (error1**2) + (error2**2)
    
    print(f"Shot 1 Error: {error1}")
    print(f"Shot 2 Error: {error2}")
    print("-" * 20)
    print(f"AI thinks total error is: {wrong_total} (Delusional! 🤡)")
    print(f"Actual AI Loss (Squared): {correct_total} (Honest! ✅)")

if __name__ == "__main__":
    test_without_squaring()
