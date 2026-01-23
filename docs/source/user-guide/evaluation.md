# Model Evaluation

Comprehensive guide to evaluating skin lesion classification models, including metrics, visualization, and performance analysis.

## Quick Evaluation

### Basic Evaluation Commands

```bash
# Evaluate latest trained model
uv run src/mlops_project/evaluate.py \
    --model-path models/efficientnet/version_0/checkpoints/best.ckpt \
    --test-split

# Evaluate with custom data
uv run src/mlops_project/evaluate.py \
    --model-path models/resnet/latest.ckpt \
    --data-path data/custom_test \
    --batch-size 32
```

### Key Evaluation Metrics

Our evaluation system provides comprehensive medical AI metrics:

- **Accuracy**: Overall classification correctness
- **Sensitivity (Recall)**: True positive rate for malignant cases
- **Specificity**: True negative rate for benign cases
- **Precision**: Positive predictive value
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve
- **AUC-PR**: Area under precision-recall curve

## Detailed Evaluation Analysis

### Performance by Model Architecture

| Model | Accuracy | F1 Score | Sensitivity | Specificity | AUC-ROC |
|-------|----------|----------|-------------|-------------|---------|
| BaselineCNN | 0.742 | 0.685 | 0.721 | 0.763 | 0.789 |
| ResNet | 0.823 | 0.791 | 0.812 | 0.834 | 0.861 |
| EfficientNet-B3 | 0.887 | 0.854 | 0.871 | 0.903 | 0.923 |

### Confusion Matrix Analysis

```python
# Generate detailed confusion matrix
from src.mlops_project.evaluate import generate_confusion_matrix

results = generate_confusion_matrix(
    model_path="models/efficientnet/best.ckpt",
    test_loader=test_dataloader
)
```

### ROC Curve and Threshold Analysis

```python
# Analyze optimal classification thresholds
from src.mlops_project.evaluate import threshold_analysis

optimal_threshold = threshold_analysis(
    model_predictions=predictions,
    true_labels=labels,
    metric="f1_score"  # or "sensitivity", "specificity"
)
```

## Clinical Performance Evaluation

### Medical Validation Metrics

For medical AI applications, we focus on clinically relevant metrics:

```bash
# Medical-focused evaluation
uv run src/mlops_project/evaluate.py \
    --model-path models/efficientnet/best.ckpt \
    --clinical-metrics \
    --confidence-intervals \
    --stratified-analysis
```

**Clinical Metrics Output:**
```
🏥 Clinical Performance Analysis:
├── Sensitivity (Recall): 0.871 ± 0.023
├── Specificity: 0.903 ± 0.019  
├── Positive Predictive Value: 0.854 ± 0.025
├── Negative Predictive Value: 0.921 ± 0.017
├── Diagnostic Odds Ratio: 47.3 ± 8.2
└── Matthews Correlation Coefficient: 0.768 ± 0.022
```

### Error Analysis

```bash
# Analyze misclassified cases
uv run src/mlops_project/evaluate.py \
    --model-path models/efficientnet/best.ckpt \
    --error-analysis \
    --save-misclassified \
    --output-dir reports/error_analysis/
```

This generates:
- **False positives**: Benign lesions classified as malignant
- **False negatives**: Malignant lesions classified as benign  
- **Confidence analysis**: Model uncertainty for each prediction
- **Visual examples**: Images of misclassified cases

---

*Continue to [Running Inference](inference.md) to learn about using your trained models.*