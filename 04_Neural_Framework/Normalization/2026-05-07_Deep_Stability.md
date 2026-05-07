# Deep Signal Stability Simulation (2026-05-07)

## 🧨 The Multiplier Effect
In a 100-layer network, a multiplier of **1.2** results in a signal growth of **~82 million**. This causes `NaN` failures.

## 🛡️ LayerNorm as a Checkpoint
LayerNorm acts as a "Reset Point" at every layer, ensuring the signal never exceeds the range defined by Gamma and Beta.

### 🔬 Simulation Results (Layer 5)
*   **Wild Std:** 52.31 (Exploded)
*   **Normed Std:** 1.20 (Stable)

**Conclusion:** Normalization is the only reason we can build "Deep" architectures like GPT/Llama.
