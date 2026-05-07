# Deep Signal Propagation and Stability

When building deep networks (like a 100-layer Transformer), the biggest enemy is **Mathematical Explosion**.

## 🧨 The Multiplier Effect (The "Increasing" Problem)
If a signal passes through many layers without normalization, even a tiny amplification at each step leads to total failure.

*   If every layer multiplies the signal by **1.2**:
    *   Layer 1: 1.2
    *   Layer 50: ~9,100
    *   Layer 100: **~82,817,000**
*   This explosion leads to `NaN` (Not a Number), causing the entire model to crash.

## 🛡️ How LayerNorm Saves the Deep Model
LayerNorm acts as a "Checkpoint" at every single layer. No matter how much a weight matrix tries to explode the signal, LayerNorm resets it before it reaches the next floor.

### 🔬 Real-Time Math Battle (5 Layers)
In simulations, we see the "Wild" path (no norm) vs the "Normed" path:

| Layer | Wild Std (Explosion) | Normed Std (Stability) |
| :--- | :--- | :--- |
| Start | 0.38 | 0.38 |
| Layer 1 | 1.31 | 1.20 |
| Layer 3 | **6.18** | 1.20 |
| Layer 5 | **52.31** | 1.20 |

**Conclusion:** Without LayerNorm, deep networks are impossible to train. With it, we can stack 100+ layers and the signal stays as calm and predictable as it was at Layer 1.
