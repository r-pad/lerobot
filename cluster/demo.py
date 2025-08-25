"""Simple demo for launching with slurm."""
import torch

if __name__ == "__main__":
    print(f"Number of GPUs: {torch.cuda.device_count()}")
