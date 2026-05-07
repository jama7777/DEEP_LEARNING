# Theory: Matrix Math (The Multi-Lane Highway)

Real AI doesn't process one person at a time; it processes a **Batch** of data.

### Vectors and Matrices
*   **Vector:** A single row of numbers (e.g., [Speed, Strength, Endurance] for one athlete).
*   **Matrix:** A table of numbers (e.g., an entire team of athletes).

### The Magic of the Transpose ($X.T$)
When calculating gradients for a whole batch, we flip the input matrix sideways ($X^T$). 
*   **Why?** Because we need to align all the "Speeds" from every athlete into one row so the Dot Product can multiply them by the Errors in one shot.
*   **The "Seat Number" Rule:** The first element in the speed row only talks to the first element in the error list. This ensures the 1st person's data only affects the 1st person's mistake.

### Summary:
Matrix math allows the AI to find a **Consensus**. It looks at everyone at once and finds the weight update that makes the whole team better, not just one individual.
