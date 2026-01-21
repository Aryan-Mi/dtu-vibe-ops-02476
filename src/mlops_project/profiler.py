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
        warmup_steps: int = 5,
        active_steps: int = 2,
        repeat_cycles: int = 1,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.schedule_every_n_steps = schedule_every_n_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.repeat_cycles = repeat_cycles
        self.profiler = None
        self._profiling_active = False

        # Increase file descriptor limit warnings
        print(f"  Profiler initialized with {repeat_cycles} cycle(s), {active_steps} active steps per cycle")
        print("  Tip: If you get 'Too many open files' error, reduce repeat_cycles or active_steps")

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
            with_stack=False,  # Disabled to avoid PyTorch bug
            with_flops=True,
            with_modules=True,
        )
        self.profiler.start()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Step the profiler after each batch."""
        if self.profiler is not None:
            try:
                self.profiler.step()
            except RuntimeError as e:
                if "stack.empty()" in str(e):
                    # Known PyTorch profiler bug - skip this step
                    if batch_idx % 10 == 0:  # Only print every 10th to avoid spam
                        print(f"  Profiler step skipped at batch {batch_idx} (known PyTorch issue)")
                else:
                    print(f"  Profiler error at batch {batch_idx}: {e}")
                    # Continue training without profiling for this step

    def on_train_end(self, trainer, pl_module):
        """Stop profiler and export Chrome trace when training ends."""
        if self.profiler is not None:
            try:
                self.profiler.stop()
            except Exception as e:  # noqa: BLE001
                print(f"  Warning: Profiler stop error (traces may already be saved): {e}")

            # Export Chrome trace for manual viewing
            chrome_trace_path = self.output_dir / "chrome_trace.json"
            try:
                if hasattr(self.profiler, "export_chrome_trace"):
                    self.profiler.export_chrome_trace(str(chrome_trace_path))
                    print(f"  Chrome trace exported to: {chrome_trace_path}")
                    print("  View in Chrome: chrome://tracing")
            except Exception as e:  # noqa: BLE001
                print("  Note: Chrome trace export skipped (already saved via TensorBoard handler)")

            # Export profiler summary
            summary_path = self.output_dir / "profiler_summary.txt"
            try:
                with open(summary_path, "w") as f:
                    if self.profiler.key_averages():
                        f.write("Top 10 operations by CPU time:\n")
                        f.write(self.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                        f.write("\n\nTop 10 operations by CUDA time:\n")
                        f.write(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                print(f"  Profiler summary saved to: {summary_path}")
            except Exception as e:  # noqa: BLE001
                print(f"  Warning: Could not save profiler summary: {e}")

            print(f"  TensorBoard logs available at: {self.output_dir}")
            print(f"  Run: tensorboard --logdir={self.output_dir}")
