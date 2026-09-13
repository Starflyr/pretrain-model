import torch


print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "先排查 GPU 版 PyTorch 与驱动，不要继续训练"
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
x = torch.randn(512, 512, device="cuda", requires_grad=True)
loss = (x @ x).square().mean()
loss.backward()
torch.cuda.synchronize()
print("GPU forward/backward OK; loss:", loss.item())
