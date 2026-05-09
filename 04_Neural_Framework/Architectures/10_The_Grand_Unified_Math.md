# 🪐 Theory Note 10: The Grand Unified Math

This note documents the "Complete Life Cycle" of a signal through our **Stable Verbal Block**.

## 1. THE FORWARD FLOW (The Life Cycle)
Every piece of data goes through 5 stages of transformation:

1.  **Stage 1: Input (x)** - The raw embedding.
2.  **Stage 2: LayerNorm (LN)** - The volume is equalized.
    *   $x_{norm} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$
3.  **Stage 3: Linear Projection (W, b)** - The "thinking" step.
    *   $z = x_{norm} \cdot W + b$
4.  **Stage 4: Activation (ReLU)** - The "gate" step.
    *   $a = \max(0, z)$
5.  **Stage 5: Residual (Add)** - The "memory" step.
    *   $y = x + a$

---

## 2. THE BACKWARD FLOW (The Gradient Superhighway)
When the error ($dy$) comes back, it splits into two paths:

### Path A: The Skip path
*   The gradient travels directly to the previous layer.
*   $dx_{skip} = dy$ (Gradient is preserved 100%).

### Path B: The Learn path
*   The gradient travels through the weights.
*   $dz = dy \cdot (z > 0)$ (ReLU Gate)
*   $dW = x_{norm}^T \cdot dz$
*   $dx_{norm} = dz \cdot W^T$

### The Merge:
*   $dx_{total} = dx_{skip} + LN\_backward(dx_{norm})$

---

## 3. THE OPTIMIZATION (The Driver)
We use the **Adam Engine** to update $W$ and $b$:
1.  **Momentum ($m$):** Remembers the direction.
2.  **Velocity ($v$):** Remembers the speed/variance.
3.  **Bias Correction:** Adjusts for the "cold start" in early training.
4.  **Final Update:** $W = W - \alpha \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$

---

## 💡 THE PRIMARY MOTTO:
**"Stability at every step."** 
LayerNorm stabilizes the **Forward** pass. Residuals stabilize the **Backward** pass. Adam stabilizes the **Update** pass. Together, they create a network that can learn anything.
