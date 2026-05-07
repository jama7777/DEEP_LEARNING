import numpy as np

class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr
    def update(self, w, g):
        return w - self.lr * g

class Adam:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0
        self.v = 0
        self.t = 0
    def update(self, w, g):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return w - self.lr * (m_hat / (np.sqrt(v_hat) + 1e-8))

def run_race(track_name, gradients, initial_w=10.0):
    print(f"\n🏁 TRACK: {track_name}")
    print("-" * 50)
    
    w_sgd, w_adam = initial_w, initial_w
    sgd = SGD(lr=0.1)
    adam = Adam(lr=0.1)
    
    for i, g in enumerate(gradients, 1):
        w_sgd = sgd.update(w_sgd, g)
        w_adam = adam.update(w_adam, g)
        
    print(f"Final SGD Position:  {w_sgd:8.4f}")
    print(f"Final Adam Position: {w_adam:8.4f}")
    
    diff = abs(w_adam - 0) # Distance from target 0
    diff_sgd = abs(w_sgd - 0)
    
    if diff < diff_sgd:
        print(f"🏆 WINNER: ADAM (closer to target by {diff_sgd - diff:.4f} units)")
    else:
        print(f"🏆 WINNER: SGD (closer to target)")

def main():
    print("🏆 THE GRAND OPTIMIZER RACE: SGD vs. ADAM")
    print("=" * 60)

    # TRACK 1: THE DRAG STRIP (Constant 1.0)
    run_race("THE DRAG STRIP (Consistent Signal)", [1.0] * 50)

    # TRACK 2: THE GHOST ROAD (Tiny Signal 0.001)
    run_race("THE GHOST ROAD (Tiny Signal)", [0.001] * 50)

    # TRACK 3: THE BUMPY ROAD (Jittery Signal +1, -1)
    # Target is to stay at 10.0 (The start)
    run_race("THE BUMPY ROAD (Noisy/Jittery)", [1.0, -0.9, 1.0, -0.9] * 12, initial_w=10.0)

if __name__ == "__main__":
    main()
