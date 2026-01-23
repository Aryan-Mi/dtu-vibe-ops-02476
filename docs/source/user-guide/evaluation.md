# Model Evaluation

## Basic Evaluation

```bash
uv run python -m mlops_project.evaluate --checkpoint models/best_model.ckpt
```

## Evaluation Metrics

The evaluation produces:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class metrics
- **F1 Score**: Harmonic mean
- **Confusion Matrix**: Prediction distribution
- **AUC-ROC**: Classification threshold analysis

## Output

Results are saved to `outputs/evaluation/`:
- `metrics.json` - Numerical results
- `confusion_matrix.png` - Visualization
- `roc_curve.png` - ROC analysis
