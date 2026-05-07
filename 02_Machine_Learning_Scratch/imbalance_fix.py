import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def generate_imbalanced_data(minority_class=0):
    np.random.seed(42)
    if minority_class == 0:
        # 997 Fever people (38.0 to 40.0) -> y=1
        X_majority = 38.0 + 2.0 * np.random.rand(997, 1)
        y_majority = np.ones((997, 1))
        # 3 Healthy people (36.0 to 37.0) -> y=0
        X_minority = 36.0 + 1.0 * np.random.rand(3, 1)
        y_minority = np.zeros((3, 1))
    else: # minority_class == 1
        # 997 Healthy people (36.0 to 37.0) -> y=0
        X_majority = 36.0 + 1.0 * np.random.rand(997, 1)
        y_majority = np.zeros((997, 1))
        # 3 Fever people (38.0 to 40.0) -> y=1
        X_minority = 38.0 + 2.0 * np.random.rand(3, 1)
        y_minority = np.ones((3, 1))
        
    X = np.vstack((X_majority, X_minority))
    y = np.vstack((y_majority, y_minority))
    return X, y

def train(X, y, weight_multiplier=1.0, minority_class=0):
    m = len(X)
    X_mean, X_std = np.mean(X), np.std(X)
    X_norm = (X - X_mean) / X_std
    
    w = np.random.randn(1, 1)
    b = np.zeros((1, 1))
    lr = 0.1
    
    for epoch in range(100001):
        prediction = sigmoid(X_norm.dot(w) + b)
        error = prediction - y
        
        # TARGET THE MINORITY CLASS
        punishment_mask = np.where(y == minority_class, weight_multiplier, 1.0)
        weighted_error = error * punishment_mask
        
        # Calculate Weighted Loss (The "Pain")
        loss = -np.mean(punishment_mask * (y * np.log(prediction + 1e-15) + (1 - y) * np.log(1 - prediction + 1e-15)))
        
        gw = (1/m) * X_norm.T.dot(weighted_error)
        gb = (1/m) * np.sum(weighted_error)
        
        w -= lr * gw
        b -= lr * gb

        if epoch % 20000 == 0:
            print(f"Epoch {epoch:6}: Loss = {loss:.6f} | Weight = {w[0][0]:.4f}")
            
    return w, b, X_mean, X_std

def main():
    print("======================================================")
    print("CASE A: 0 is the minority (3 Healthy, 997 Sick)")
    print("======================================================")
    X_A, y_A = generate_imbalanced_data(minority_class=0)
    test_temp_healthy = 28 # Healthy person
    
    print("\n--- CASE A - TRIAL 1: No Punishment (Ignores the 3 healthy people) ---")
    w1, b1, m1, s1 = train(X_A, y_A, weight_multiplier=1.0, minority_class=0)
    pred1 = sigmoid(((test_temp_healthy - m1) / s1) * w1 + b1)
    print(f"Confidence for {test_temp_healthy} degrees: {pred1[0][0]:.4f}")
    print("Result: " + ("FEVER (Error! It should be Normal)" if pred1 > 0.5 else "NORMAL"))

    print("\n--- CASE A - TRIAL 2: Punish missing the 0s (Targeted punishment) ---")
    w2, b2, m2, s2 = train(X_A, y_A, weight_multiplier=500.0, minority_class=0)
    pred2 = sigmoid(((test_temp_healthy - m2) / s2) * w2 + b2)
    print(f"Confidence for {test_temp_healthy} degrees: {pred2[0][0]:.4f}")
    print("Result: " + ("FEVER" if pred2 > 0.5 else "NORMAL (Success!)"))

    print("\n\n======================================================")
    print("CASE B: 1 is the minority (3 Sick, 997 Healthy)")
    print("======================================================")
    X_B, y_B = generate_imbalanced_data(minority_class=1)
    test_temp_sick = 42 # Very sick person
    
    print("\n--- CASE B - TRIAL 1: No Punishment (Ignores the 3 sick people) ---")
    w3, b3, m3, s3 = train(X_B, y_B, weight_multiplier=1.0, minority_class=1)
    pred3 = sigmoid(((test_temp_sick - m3) / s3) * w3 + b3)
    print(f"Confidence for {test_temp_sick} degrees: {pred3[0][0]:.4f}")
    print("Result: " + ("FEVER" if pred3 > 0.5 else "NORMAL (Error! It should be Fever)"))

    print("\n--- CASE B - TRIAL 2: Punish missing the 1s (Targeted punishment) ---")
    w4, b4, m4, s4 = train(X_B, y_B, weight_multiplier=500.0, minority_class=1)
    pred4 = sigmoid(((test_temp_sick - m4) / s4) * w4 + b4)
    print(f"Confidence for {test_temp_sick} degrees: {pred4[0][0]:.4f}")
    print("Result: " + ("FEVER (Success!)" if pred4 > 0.5 else "NORMAL"))

if __name__ == "__main__":
    main()
