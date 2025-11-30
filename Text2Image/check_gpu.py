import torch

try:
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print("✅ Success! GPU Detected.")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("Your RTX 3090 is ready for Deep Learning! 🚀")
    else:
        print("❌ Error: GPU NOT detected. Running on CPU.")
except ImportError:
    print("❌ Error: PyTorch is not installed correctly.")