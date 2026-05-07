import numpy as np
from xray_utils import show_detailed_math, show_dot_logic, show_activation_logic

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    # 1. INPUT DATA (One athlete: Speed, Strength, Endurance)
    X = np.array([0.9, 0.8, 0.2])
    
    # 2. THE HIDDEN LAYER (4 Neurons)
    W_hidden = np.array([
        [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [-0.1, 0.5, 0.1]
    ])
    b_hidden = np.array([0.1, 0.2, 0.3, 0.4])
    
    print("--- FORWARD PROP: X-RAY EDITION ---")
    
    # 3. Step 1: Input to Hidden
    hidden_scores = np.dot(W_hidden, X) + b_hidden
    show_detailed_math("1. HIDDEN SCORES (W_hidden * X + b)", [W_hidden, X, b_hidden], hidden_scores)
    show_dot_logic("HiddenScore[0]", W_hidden, X, row=0)
    
    hidden_outputs = sigmoid(hidden_scores)
    show_activation_logic("HIDDEN ACTIVATIONS", hidden_scores, hidden_outputs, "sigmoid")
    
    # 4. Step 2: Hidden to Output
    W_output = np.array([0.2, 0.3, 0.4, 0.5])
    b_output = 0.1
    
    final_score = np.dot(W_output, hidden_outputs) + b_output
    show_detailed_math("2. FINAL SCORE (W_output * A_hidden + b)", [W_output, hidden_outputs, b_output], final_score)
    show_dot_logic("FinalScore", W_output, hidden_outputs)
    
    final_prediction = sigmoid(final_score)
    show_activation_logic("FINAL PREDICTION", final_score, final_prediction, "sigmoid")

if __name__ == "__main__":
    main()
