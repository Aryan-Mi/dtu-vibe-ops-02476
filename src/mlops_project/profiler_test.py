import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from src.mlops_project.model import BaselineCNN


def profiling_test():
    """Profiling with a simple forward pass."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create a simple model (like your BaselineCNN)
    model = BaselineCNN(
        num_classes=7,
        input_dim=224,
        input_channel=3,
        output_channels=[16, 32],
        lr=1e-3,
    ).to(device)

    # Create dummy data
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_target = torch.randint(0, 7, (batch_size,), device=device)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Target shape: {dummy_target.shape}")

    # Setup profiler output directory
    output_dir = Path("demo_profiler_logs")
    output_dir.mkdir(exist_ok=True)

    # Create profiler with similar config to your training
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    print("\nStarting profiling...")

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler(str(output_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        # Run multiple iterations like in training
        for step in range(10):
            print(f"Step {step + 1}/10", end="\r")
            # Forward pass
            with torch.set_grad_enabled(True):
                logits = model(dummy_input)
                loss = torch.nn.functional.cross_entropy(logits, dummy_target)
            # Backward pass
            loss.backward()
            # Optimizer step (simulated)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Step the profiler
            prof.step()
            # Clear gradients
            model.zero_grad()
            # Small delay to simulate real training
            time.sleep(0.01)

    print(f"Profiler logs saved to: {output_dir}")

    # Print profiling summary
    print("\nSummary:")
    print("-" * 30)

    # Top operations by CPU time
    print("\nTop 5 operations by CPU time:")
    cpu_stats = prof.key_averages().table(sort_by="cpu_time_total", row_limit=5)
    print(cpu_stats)

    # Memory usage
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        gpu_stats = prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=5)
        print(gpu_stats)

    # TensorBoard instructions
    print("\nView detailed results:")
    print(f"   tensorboard --logdir={output_dir}")
    print("   Then navigate to: http://localhost:6006/#pytorch_profiler")

    # Chrome trace instructions
    chrome_trace = output_dir / "chrome_trace.json"
    if chrome_trace.exists():
        print("\n Chrome Trace:")
        print(f"   File: {chrome_trace}")
        print("   Open: chrome://tracing")
        print("   Load the JSON file to see timeline view")


if __name__ == "__main__":
    profiling_test()
