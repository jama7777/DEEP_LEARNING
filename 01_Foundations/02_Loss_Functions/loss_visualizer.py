import numpy as np

def calculate_mse(predicted, actual):
    # Loss = (Pred - Actual)^2
    return (predicted - actual) ** 2

def main():
    print("--- AI Error (Loss) Calculator ---")
    
    actual_price = 15000  # The goal
    predictions = [14000, 16000, 15100, 25000] # Different AI guesses
    
    print(f"Target Value: {actual_price}\n")
    print(f"{'Prediction':<12} | {'Raw Error':<12} | {'MSE Loss':<12}")
    print("-" * 45)
    
    for pred in predictions:
        error = pred - actual_price
        loss = calculate_mse(pred, actual_price)
        
        print(f"{pred:<12} | {error:<12} | {loss:<12,}")

    print("\nTakeaway:")
    print("1. Notice how -1000 and +1000 error both result in the same positive Loss (1,000,000).")
    print("2. Notice how the error of 10,000 (last row) resulted in a massive Loss of 100,000,000!")
    print("   The square makes the AI 'scared' of large mistakes.")

if __name__ == "__main__":
    main()
