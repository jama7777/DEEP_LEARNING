# How Sigmoid and Tanh "Squash" Numbers

The magic behind why these functions never go above or below their limits lies in **Exponentials ($e^x$)**.

## 1. The Sigmoid Math: Why 0 to 1?

The formula is:
$$S(x) = \frac{1}{1 + e^{-x}}$$

Let's see what happens to the denominator:

### Case A: Input is a Huge Positive Number ($x = 100$)
1. $e^{-100}$ is a tiny, tiny number (almost zero).
2. The formula becomes: $\frac{1}{1 + 0.00000...} \approx \frac{1}{1}$
3. **Result: 1.0** (It can never go higher than 1 because the denominator will always be at least 1).

### Case B: Input is a Huge Negative Number ($x = -100$)
1. $e^{-(-100)}$ becomes $e^{100}$, which is an astronomical number.
2. The formula becomes: $\frac{1}{1 + \text{Trillions}} \approx \frac{1}{\text{Infinity}}$
3. **Result: 0.0** (It can never go lower than 0 because you're dividing 1 by a positive number).

---

## 2. The Tanh Math: Why -1 to 1?

The formula is:
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

It's basically comparing two opposing forces.

### Case A: Huge Positive Input ($x = 100$)
1. $e^{100}$ is massive, $e^{-100}$ is zero.
2. Formula: $\frac{\text{Massive} - 0}{\text{Massive} + 0} = 1$
3. **Result: 1.0**

### Case B: Huge Negative Input ($x = -100$)
1. $e^{-100}$ is zero, $e^{100}$ is massive.
2. Formula: $\frac{0 - \text{Massive}}{0 + \text{Massive}} = -1$
3. **Result: -1.0**

---

## Summary Table of "The Limits"

| Function | If $x$ is HUGE (+) | If $x$ is HUGE (-) | Why? |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | Hits $1.0$ | Hits $0.0$ | Denominator grows to infinity or shrinks to 1 |
| **Tanh** | Hits $1.0$ | Hits $-1.0$ | Numerator and Denominator become equal (but sign changes) |

> [!TIP]
> This is called **Asymptotic Behavior**. The curve gets closer and closer to the line but never actually "breaks" through it. No matter how hard you push the score (even $1,000,000$), Sigmoid will just give you $0.999999...$
