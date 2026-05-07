# Theory: The Calculus of AI

Machine Learning relies on one specific rule of calculus: **The Power Rule.**

### Why the "2" in $(error)^2$?
If Error ($e$) is the side of a square, the Loss is the **Area** ($e^2$). If you increase the error slightly, the area grows by **two strips** ($2 \cdot e \cdot de$). 

That is why the "2" appears—it represents the two growing edges of the error square. Mathematically, it is the exact rate at which the error area is expanding.

### The "First Principles" Derivation:
If we change the weight $w$ by a tiny amount $h$:
1.  **Before:** $(wx - t)^2$
2.  **After:** $((w+h)x - t)^2$
3.  **Difference:** $2(wx - t)hx + h^2x^2$
4.  **Rate (Divide by h):** $2(wx - t)x + hx^2$
5.  **Limit (h goes to 0):** $2 \cdot (wx - t) \cdot x$

### Summary:
*   The **2** comes from the Square shape.
*   The **(wx - t)** is the size of the Gap.
*   The **x** is the Leverage of the input.
