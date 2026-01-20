from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


class ProfilerCallback(pl.Callback):
    """Callback for advanced PyTorch profiling with Chrome trace export."""

    def __init__(
        self,
        output_dir: str,
        schedule_every_n_steps: int = 50,
        warmup_steps: int = 1,
        active_steps: int = 3,
        repeat_cycles: int = 2,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.schedule_every_n_steps = schedule_every_n_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.repeat_cycles = repeat_cycles
        self.profiler = None

    def on_train_start(self, trainer, pl_module):
        """Initialize PyTorch profiler when training starts."""
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self.profiler = profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.warmup_steps,
                warmup=self.warmup_steps,
                active=self.active_steps,
                repeat=self.repeat_cycles,
                skip_first=self.warmup_steps,
            ),
            on_trace_ready=tensorboard_trace_handler(str(self.output_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )
        self.profiler.start()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Step the profiler after each batch."""
        if self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, trainer, pl_module):
        """Stop profiler and export Chrome trace when training ends."""
        if self.profiler is not None:
            self.profiler.stop()

            # Export Chrome trace for manual viewing
            chrome_trace_path = self.output_dir / "chrome_trace.json"
            if hasattr(self.profiler, "export_chrome_trace"):
                self.profiler.export_chrome_trace(str(chrome_trace_path))
                print(f"  Chrome trace exported to: {chrome_trace_path}")
                print("  View in Chrome: chrome://tracing")

            # Export profiler summary
            summary_path = self.output_dir / "profiler_summary.txt"
            with open(summary_path, "w") as f:
                if self.profiler.key_averages():
                    f.write("Top 10 operations by CPU time:\n")
                    f.write(self.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                    f.write("\n\nTop 10 operations by CUDA time:\n")
                    f.write(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            print(f"  Profiler summary saved to: {summary_path}")

            print(f"  TensorBoard logs available at: {self.output_dir}")
            print(f"  Run: tensorboard --logdir={self.output_dir}")
