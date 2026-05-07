import numpy as np
import sympy as sp
from sympy import Matrix, pprint, Eq, Symbol, MatMul, MatAdd
import matplotlib.pyplot as plt

def show_elementwise_logic(name, A, B, result, op_symbol, row=0, col=0):
    val_a = A[row, col]; val_b = B[row, col]; res = result[row, col]
    print(f"\n🧩 ELEMENTWISE LOGIC: How we get {name}[{row},{col}]")
    print(f"Calculation: {val_a:.4f} {op_symbol} {val_b:.4f} = {res:.4f}")

def show_activation_logic(name, input_data, result, func_name, row=0, col=0):
    # Handle both scalar and matrix inputs
    if np.isscalar(input_data):
        val_in = input_data; res = result
    else:
        val_in = input_data[row, col]; res = result[row, col]
    print(f"\n✨ ACTIVATION LOGIC: How we get {name}")
    print(f"Calculation: {func_name}({val_in:.4f}) = {res:.4f}")

def show_dot_logic(name, A, B, row=0, col=0):
    # Handle 1D or 2D
    if A.ndim == 1: A_row = A
    else: A_row = A[row, :]
    
    if B.ndim == 1: B_col = B
    else: B_col = B[:, col]
    
    terms = [f"({r:.2f} * {c:.2f})" for r, c in zip(A_row, B_col)]
    calc_str = " + ".join(terms); result = np.dot(A_row, B_col)
    print(f"\n🔍 DOT PRODUCT LOGIC: How we get {name}")
    print(f"Calculation: {calc_str} = {result:.4f}")

def show_detailed_math(title, components, result, operation="*", label=None):
    print(f"\n--- {title} ---")
    
    # Convert all components to sympy Matrix (handling scalars too)
    parts = []
    for item in components:
        if np.isscalar(item): parts.append(Matrix([[np.round(item, 2)]]))
        else: parts.append(Matrix(np.round(item, 2)))
        
    if np.isscalar(result): res_m = Matrix([[np.round(result, 2)]])
    else: res_m = Matrix(np.round(result, 2))
    
    if len(parts) == 1:
        expr = Symbol(label) if label else parts[0]
    elif len(parts) == 2:
        if operation == "*": expr = MatMul(parts[0], parts[1], evaluate=False)
        elif operation == "-": expr = MatAdd(parts[0], -parts[1], evaluate=False)
        elif operation == "+": expr = MatAdd(parts[0], parts[1], evaluate=False)
        else: expr = res_m
    elif len(parts) == 3:
        bias = parts[2]
        # Broadcasting logic for SymPy
        if bias.shape[0] == 1 and parts[0].shape[0] > 1:
            bias = Matrix([bias.row(0)] * parts[0].shape[0])
        expr = MatAdd(MatMul(parts[0], parts[1], evaluate=False), bias, evaluate=False)
    
    pprint(Eq(expr, res_m, evaluate=False))

def visualize_network(X, W1, W2, h_act, preds, sample_idx=0):
    """
    Draws a premium neural network diagram showing the values for a specific sample.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_axis_off()
    
    # Layer 1: Inputs (2)
    # Layer 2: Hidden (4)
    # Layer 3: Output (1)
    layer_sizes = [2, 4, 1]
    v_spacing = 1.0
    h_spacing = 2.5
    
    node_coords = []
    
    # Draw Nodes
    for i, size in enumerate(layer_sizes):
        layer_coords = []
        x = i * h_spacing
        top = (size - 1) * v_spacing / 2
        for j in range(size):
            y = top - j * v_spacing
            layer_coords.append((x, y))
            
            # Draw Circle
            color = '#00d4ff' if i == 0 else ('#39ff14' if i == 1 else '#ff007f')
            circle = plt.Circle((x, y), 0.15, color=color, alpha=0.8, zorder=4)
            ax.add_artist(circle)
            
            # Label values inside/near nodes
            if i == 0: val = X[sample_idx, j]
            elif i == 1: val = h_act[sample_idx, j]
            else: val = preds[sample_idx, j]
            ax.text(x, y + 0.25, f"{val:.2f}", color='white', ha='center', fontweight='bold', fontsize=10)
            
        node_coords.append(layer_coords)

    # Draw Connections (Weights)
    # W1 is (2, 4), W2 is (4, 1)
    weights = [W1, W2]
    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i+1]):
                start = node_coords[i][j]
                end = node_coords[i+1][k]
                
                # Weight value
                w_val = weights[i][j, k]
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                
                # Draw Line
                ax.plot([start[0], end[0]], [start[1], end[1]], color='white', alpha=0.2, linewidth=abs(w_val)*2, zorder=1)
                
                # Draw weight label
                if abs(w_val) > 0.1: # Only show significant weights to avoid clutter
                    ax.text(mid_x, mid_y + (k-j)*0.05, f"{w_val:.2f}", color='#ffff00', fontsize=8, ha='center', alpha=0.6)

    plt.title(f"Neural Network Internal State (Sample {sample_idx})", fontsize=16, fontweight='bold', color='white', pad=20)
    
    save_path = f'/Users/indra/Desktop/DEEP_LEARNING/03_Neural_Networks_Scratch/network_state_sample_{sample_idx}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\n🎨 Neural Visualization saved to: {save_path}")
    plt.close()
