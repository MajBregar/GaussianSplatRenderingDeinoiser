# check_torch.py

import os
import sys


def main():
    print("Python version:")
    print(sys.version)
    print()

    try:
        import torch
        print("PyTorch imported successfully")
        print(f"torch version: {torch.__version__}")
    except Exception as e:
        print("PyTorch import failed")
        print(repr(e))
        return 1

    try:
        import torchvision
        print("torchvision imported successfully")
        print(f"torchvision version: {torchvision.__version__}")
    except Exception as e:
        print("torchvision import failed")
        print(repr(e))
        return 1

    print()
    print("Backend check:")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.version.cuda: {torch.version.cuda}")

    if hasattr(torch.version, "hip"):
        print(f"torch.version.hip: {torch.version.hip}")

    if torch.cuda.is_available():
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of CUDA/HIP devices: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

        device = torch.device("cuda")
    else:
        print("CUDA/HIP is not available. Testing CPU only.")
        device = torch.device("cpu")

    print()
    print("Tensor operation check:")
    try:
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        z = x @ y

        if device.type == "cuda":
            torch.cuda.synchronize()

        print("Tensor multiplication succeeded")
        print(z)
    except Exception as e:
        print("Tensor operation failed")
        print(repr(e))
        return 1

    print()
    print("Autograd check:")
    try:
        a = torch.randn(5, requires_grad=True, device=device)
        loss = (a ** 2).sum()
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        print("Autograd succeeded")
        print(f"Gradient: {a.grad}")
    except Exception as e:
        print("Autograd failed")
        print(repr(e))
        return 1

    print()
    print("torchvision model check:")
    try:
        model = torchvision.models.resnet18(weights=None)
        model = model.to(device)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224, device=device)

        with torch.no_grad():
            output = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        print("torchvision ResNet18 forward pass succeeded")
        print(f"Output shape: {tuple(output.shape)}")
    except Exception as e:
        print("torchvision model test failed")
        print(repr(e))
        return 1

    print()
    print("Cleanup check:")
    try:
        del x, y, z
        del a, loss
        del model, dummy_input, output

        if device.type == "cuda":
            print("Synchronizing GPU...")
            torch.cuda.synchronize()

            print("Clearing GPU cache...")
            torch.cuda.empty_cache()

        print("Cleanup completed")
    except Exception as e:
        print("Cleanup failed")
        print(repr(e))
        return 1

    print()
    print("All checks passed.")
    print("Script reached normal end.")

    return 0


if __name__ == "__main__":
    exit_code = main()
    FORCE_EXIT_AFTER_SUCCESS = False

    if FORCE_EXIT_AFTER_SUCCESS and exit_code == 0:
        os._exit(0)

    sys.exit(exit_code)