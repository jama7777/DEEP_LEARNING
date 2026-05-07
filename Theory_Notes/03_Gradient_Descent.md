# Theory: Gradient Descent & Weight Updates

Gradient Descent is the process of finding the bottom of a "Loss Bowl."

### The Compass (The Gradient)
The Gradient tells the AI two things:
1.  **Direction:** Are we facing Uphill (+) or Downhill (-)?
2.  **Magnitude:** How steep is the hill? (How far are we from the bottom?)

### The Architect vs. The Brakes
In a weight update `w = w - (lr * gradient)`:
*   **The Input ($x$) is the Architect:** It decides the **Ratio** of the update. It tells the AI which specific weights are "to blame" for the error.
*   **The Learning Rate ($lr$) is the Brakes:** It decides the **Scale**. It ensures the AI doesn't take a "Hulk Jump" over the valley and land on the other side (Overshooting).

### The Law of Responsibility:
Without the input in the math, the AI would update all weights equally, which would be a disaster. The input is the only way the AI knows **which** knob to turn.
