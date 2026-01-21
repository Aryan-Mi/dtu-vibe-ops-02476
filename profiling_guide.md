# Profiling Guide

This guide explains how to use the profiling functionality that has been set up for your PyTorch Lightning training.

## What was added

1. **PyTorch Profiler Integration**: Advanced profiling using `torch.profiler` with TensorBoard integration
2. **Configuration Options**: Profiler settings in `configs/config.yaml`
3. **Chrome Trace Export**: Ability to export traces for Chrome DevTools visualization
4. **Automatic Callback**: ProfilerCallback handles profiling lifecycle automatically

## How to enable profiling

### Method 1: Command Line
```bash
# Enable profiling with default settings
uv run src/mlops_project/train.py profiling.enabled=true

# With specific model and subsampling
uv run src/mlops_project/train.py profiling.enabled=true model=resnet data.subsample_percentage=0.1
```

### Method 2: Configuration
Edit `configs/config.yaml`:
```yaml
profiling:
  enabled: true
  output_dir: "outputs/profiler_logs"
  schedule_every_n_steps: 50
  wait_steps: 1
  warmup_steps: 1
  active_steps: 3
  repeat_cycles: 2
```

## Profiling Output

When profiling is enabled, you'll get:

### 1. Console Output
- Basic PyTorch Lightning profiler report showing time breakdown
- Identifies bottlenecks like `run_training_epoch`, `optimizer_step`, etc.

### 2. TensorBoard Logs
- Location: `outputs/profiler_logs/{model_name}/`
- Files: `*.pt.trace.json`
- View with: `tensorboard --logdir=outputs/profiler_logs/{model_name}/`

### 3. Chrome Trace Export
- File: `outputs/profiler_logs/{model_name}/chrome_trace.json`
- Open Chrome, go to `chrome://tracing`
- Load the JSON file to see detailed timeline view

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|----------|
| `enabled` | Enable/disable profiling | `false` |
| `output_dir` | Directory for profiler outputs | `"outputs/profiler_logs"` |
| `schedule_every_n_steps` | How often to profile | `50` |
| `wait_steps` | Warm-up steps before profiling | `1` |
| `warmup_steps` | Warm-up steps per cycle | `1` |
| `active_steps` | Active profiling steps per cycle | `3` |
| `repeat_cycles` | Number of profiling cycles | `2` |

## Usage Examples

### Quick Performance Check
```bash
# Fast profiling with minimal data
uv run src/mlops_project/train.py \
  profiling.enabled=true \
  data.subsample_percentage=0.01 \
  training.max_epochs=2 \
  model=baseline_cnn
```

### Comprehensive Analysis
```bash
# Full profiling with more cycles
uv run src/mlops_project/train.py \
  profiling.enabled=true \
  profiling.repeat_cycles=3 \
  profiling.active_steps=5 \
  profiling.schedule_every_n_steps=25
```

## Analyzing Results

### TensorBoard
1. Start TensorBoard:
   ```bash
   tensorboard --logdir=outputs/profiler_logs/BaselineCNN
   ```

2. Navigate to `http://localhost:6006/#pytorch_profiler`

3. View:
   - **Overview**: Performance summary
   - **Kernel**: GPU kernel performance
   - **Memory**: Memory usage patterns
   - **Trace**: Detailed timeline

### Chrome DevTools
1. Open Chrome
2. Go to `chrome://tracing`
3. Load `chrome_trace.json`
4. Explore timeline view with zoom/pan functionality

### Key Metrics to Watch
- **Training Step Time**: How long each batch takes
- **Data Loading Time**: Time spent in dataloader
- **Forward Pass**: Model inference time
- **Backward Pass**: Gradient computation time
- **Optimizer Step**: Parameter update time

## Tips for Effective Profiling

1. **Profile Multiple Runs**: Different data sizes/batch sizes
2. **Compare Models**: Profile BaselineCNN vs ResNet vs EfficientNet
3. **Focus on Bottlenecks**: Look for operations taking the most time
4. **Memory Analysis**: Check for memory leaks or spikes
5. **GPU Utilization**: Ensure you're getting good GPU usage

## Troubleshooting

### "stack.empty()" Error (FIXED)
This was a known PyTorch profiler bug that has been fixed by setting `with_stack=False` in the profiler configuration. The profiling now works correctly without this error.

### "Trace is already saved" Warning
This harmless warning appears at the end of training because TensorBoard already saves traces automatically. The training completes successfully and all profiling data is available.

### No GPU Profiling
Ensure CUDA is available:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Performance Impact
Profiling adds ~5-15% overhead. For production runs, set `profiling.enabled=false`.

## Example Performance Insights

From the sample run, key bottlenecks were:
1. **run_training_epoch**: 51.9% of total time
2. **Data loading**: 6.7% for train_dataloader_next
3. **Optimizer steps**: 4.47% for optimizer_step

Use these insights to focus optimization efforts!