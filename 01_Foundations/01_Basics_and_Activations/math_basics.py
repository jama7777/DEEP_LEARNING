import numpy as np

def manual_dot_product(v1, v2):
    """Calculate dot product without using any libraries."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of same length")
    return sum(x * y for x, y in zip(v1, v2))

def main():
    # Define two vectors
    a = [1, 2, 3]
    b = [4, 5, 6]
    
    print(f"Vector A: {a}")
    print(f"Vector B: {b}")
    
    # Manual calculation
    manual_res = manual_dot_product(a, b)
    print(f"Manual Dot Product: {manual_res}")
    
    # NumPy calculation
    np_a = np.array(a)
    np_b = np.array(b)
    np_res = np.dot(np_a, np_b)
    print(f"NumPy Dot Product: {np_res}")
    
    # Matrix Multiplication
    matrix_a = np.array([[1, 2], [3, 4]])
    matrix_b = np.array([[5, 6], [7, 8]])
    
    print("\nMatrix A:\n", matrix_a)
    print("Matrix B:\n", matrix_b)
    
    matrix_res = np.matmul(matrix_a, matrix_b)
    print("Matrix Multiplication (A @ B):\n", matrix_res)

if __name__ == "__main__":
    main()
