import numpy as np

if __name__ == "__main__":
    data = np.savez(
        "/data/example_submission.npz",
        representations=np.random.rand(20250, 192),
    )