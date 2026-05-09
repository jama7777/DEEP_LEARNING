import numpy as np

class LayerNorm:
    def __init__(self, features, eps=1e-5):
        # The 'Freedom Knobs' (Learnable parameters)
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        self.eps = eps
        
        # Cache for backprop
        self.cache = {}

    def forward(self, x):
        """
        x shape: (batch_size, features)
        """
        # 1. Calculate Mean (per sample)
        mean = np.mean(x, axis=-1, keepdims=True)
        
        # 2. Calculate Variance (per sample)
        var = np.var(x, axis=-1, keepdims=True)
        
        # 3. Standardize (The Equalization)
        x_centered = x - mean
        std = np.sqrt(var + self.eps)
        x_norm = x_centered / std
        
        # 4. Scale and Shift (The Freedom)
        out = self.gamma * x_norm + self.beta
        
        # Save for backward pass
        self.cache = {
            'x': x,
            'x_norm': x_norm,
            'x_centered': x_centered,
            'std': std,
            'mean': mean,
            'var': var
        }
        
        return out

    def backward(self, dout):
        """
        dout shape: (batch_size, features)
        Returns: dx, dgamma, dbeta
        """
        x_norm = self.cache['x_norm']
        x_centered = self.cache['x_centered']
        std = self.cache['std']
        batch_size, features = dout.shape

        # 1. Gradients for Gamma and Beta (Summed across the batch)
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        # 2. Gradient for the Input (dx) - The Chain Rule Masterpiece
        # This looks scary, but it's just the derivative of (x - mean) / std
        dx_norm = dout * self.gamma
        
        # Part A: Gradient through the variance
        dvar = np.sum(dx_norm * x_centered * -0.5 * (std**-3), axis=-1, keepdims=True)
        
        # Part B: Gradient through the mean
        dmean = np.sum(dx_norm * -1.0 / std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * x_centered, axis=-1, keepdims=True)
        
        # Part C: Combine everything for final dx
        dx = (dx_norm / std) + (dvar * 2.0 * x_centered / features) + (dmean / features)
        
        return dx, dgamma, dbeta

def microscope_test():
    print("🔬 THE LAYERNORM GRADIENT MICROSCOPE")
    print("=" * 60)
    
    # 1. Setup
    ln = LayerNorm(features=4)
    x = np.array([[1.0, 2.0, 3.0, 10.0]]) # One sample with a 'crazy' outlier
    
    # 2. Forward
    out = ln.forward(x)
    print(f"Input:  {x}")
    print(f"Output: {out}")
    print(f"Mean (After LN): {np.mean(out):.2f} (Should be 0)")
    print(f"Std  (After LN): {np.std(out):.2f} (Should be 1)")

    # 3. Dummy Gradient (Simulating error from next layer)
    dout = np.array([[0.1, -0.2, 0.3, -0.1]])
    dx, dgamma, dbeta = ln.backward(dout)

    print("\n--- 🏁 BACKPROP RESULTS ---")
    print(f"dGamma: {dgamma} (How to change volume)")
    print(f"dBeta:  {dbeta} (How to change shift)")
    print(f"dX:     {dx}")
    print("\nDeep Point: Notice how dX is small? LayerNorm 'tames' the gradients,")
    print("preventing them from becoming massive even when the input has outliers.")

if __name__ == "__main__":
    microscope_test()
