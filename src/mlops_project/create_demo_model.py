import torch
from pathlib import Path
from mlops_project.model import BaselineCNN

def main():
    print("Creating dummy demo model (random weights)...")
    
    # Initialize model with 7 classes (matching HAM10000 dataset)
    # Using BaselineCNN as it is lightweight
    model = BaselineCNN(num_classes=7, input_dim=224)
    model.eval()
    
    # Dummy input tensor [batch, channels, height, width]
    # Standard size for our pipeline is 224x224 RGB
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Output path
    output_path = Path("models/demo_model.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to {output_path}...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )
    
    print(f"✅ Demo model created successfully at: {output_path}")
    print("⚠️  Note: This model has random weights and will produce random predictions.")
    print("    Use this for testing the API/Frontend pipeline without downloading the full dataset.")

if __name__ == "__main__":
    main()
