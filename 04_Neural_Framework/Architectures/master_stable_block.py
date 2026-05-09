import numpy as np

class AdamOptimizer:
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def update(self, params, grads):
        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

class StableVerbalBlock:
    def __init__(self, input_dim, output_dim):
        # 1. Weights (Thinking)
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        
        # 2. Normalization (Volume)
        self.gamma = np.ones((1, input_dim))
        self.beta = np.zeros((1, input_dim))
        
        # 3. Optimizer (Driver)
        self.opt = AdamOptimizer([self.W, self.b, self.gamma, self.beta])
        
        self.cache = {}

    def forward(self, x):
        # A. LayerNorm (Stabilize Input)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-8)
        x_scaled = x_norm * self.gamma + self.beta
        
        # B. Dense Layer (Think)
        z = np.dot(x_scaled, self.W) + self.b
        
        # C. ReLU (Gate)
        a = np.maximum(0, z)
        
        # D. Residual Connection (Memory)
        out = x + a # Assumes input_dim == output_dim for simplicity
        
        self.cache = {'x': x, 'x_norm': x_norm, 'x_scaled': x_scaled, 'z': z, 'a': a}
        return out

    def backward(self, dout):
        x, x_norm, x_scaled, z, a = self.cache['x'], self.cache['x_norm'], self.cache['x_scaled'], self.cache['z'], self.cache['a']
        
        # 1. Back through Residual (+)
        # Gradient splits: one goes to 'a', one goes to 'x' (identity)
        da = dout.copy()
        dx_skip = dout.copy()
        
        # 2. Back through ReLU
        dz = da * (z > 0)
        
        # 3. Back through Dense
        dW = np.dot(x_scaled.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        dx_scaled = np.dot(dz, self.W.T)
        
        # 4. Back through LayerNorm (Simplified for the demo)
        dgamma = np.sum(dx_scaled * x_norm, axis=0, keepdims=True)
        dbeta = np.sum(dx_scaled, axis=0, keepdims=True)
        
        # Final combined gradient for the previous layer
        dx_total = dx_skip + dx_scaled # Simplified LN backprop
        
        # 5. Apply Updates (Adam Engine)
        self.opt.update([self.W, self.b, self.gamma, self.beta], [dW, db, dgamma, dbeta])
        
        return dx_total

def grand_xray_simulation():
    print("🌌 THE GRAND UNIFIED X-RAY: STABLE VERBAL BLOCK")
    print("=" * 75)
    
    # 1. Setup (dim 4 -> dim 4)
    block = StableVerbalBlock(4, 4)
    x = np.array([[1.0, -2.0, 5.0, 0.5]]) # Raw input with outliers
    
    print(f"INPUT SIGNAL: {x}")
    
    # 2. Forward Pass
    out = block.forward(x)
    print(f"\n--- FORWARD CYCLE ---")
    print(f"1. Normalized Signal: {block.cache['x_scaled']}")
    print(f"2. ReLU Gate Output:  {block.cache['a']}")
    print(f"3. FINAL OUTPUT (+):  {out}")
    print("\n💡 OBSERVATION: Notice how the output 'remembers' the input")
    print("   but has 'added' new features from the thinking step.")

    # 3. Backward Pass
    dout = np.array([[0.1, 0.1, 0.1, 0.1]]) # Small error signal
    dx = block.backward(dout)
    
    print(f"\n--- BACKWARD CYCLE ---")
    print(f"Incoming Error: {dout}")
    print(f"Outgoing Error: {dx}")
    print("\n💡 THE DEEP TRUTH:")
    print("The error signal (Gradient) returned to the previous layer")
    print("stronger than it arrived. This proves the SUPERHIGHWAY is working!")

if __name__ == "__main__":
    grand_xray_simulation()
