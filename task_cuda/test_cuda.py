import sys
import os
import torch

try:
    import CUDA_Test
    print("Successfully imported CUDA_Test module")
except ImportError as e:
    print(f"Failed to import CUDA_Test module: {e}")
    sys.exit(1)


def test_kernel():
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")

    A = torch.rand(128, dtype=torch.float32, device=device)
    B = torch.rand(128, dtype=torch.float32, device=device)
    try:
        result = CUDA_Test.test_kernel(A, B)
        print(f"Output result: {result.tolist()}")

        expected = A+B
        print(f"Expected result: {expected.tolist()}")

        if torch.allclose(result, expected):
            print("Test passed! Result is correct")
        else:
            print("Test failed! Result is not matching")

    except Exception as e:
        print(f"Failed to call test_kernel: {e}")


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    test_kernel()

    print("\nTest completed!")
