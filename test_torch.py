import torch
import time

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if not torch.cuda.is_available():
    print("❌ CUDA GPU is not available!")
    exit()

print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda")

print("\nCreating tensors on GPU...")

x = torch.randn(5000, 5000, device=device)
y = torch.randn(5000, 5000, device=device)

torch.cuda.synchronize()

start = time.time()

z = x @ y

torch.cuda.synchronize()

end = time.time()

print("Result device:", z.device)
print("GPU memory allocated:", round(torch.cuda.memory_allocated() / 1024**2, 2), "MB")
print("Matrix multiplication time:", round(end - start, 4), "seconds")

print("\n✅ GPU TEST PASSED!")