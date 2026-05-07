# LayerNorm: Mathematical Mechanics (2026-05-07)

## 1. The Stability Phase (The Reset)
$$\hat{x} = \frac{x - \mu}{\sigma + \epsilon}$$
*   **Mean ($\mu$):** Centering the data at zero.
*   **Std Dev ($\sigma$):** Standardizing the volume to 1.0.

## 2. The Expression Phase (Gamma & Beta)
$$y = (\hat{x} \cdot \gamma) + \beta$$
*   **Gamma ($\gamma$):** The **Scaling Knob**.
*   **Beta ($\beta$):** The **Shift Knob**.

## 💡 The Deep Truth
Stability (Reset) + Expression (Freedom) = Trainable Deep Networks.
