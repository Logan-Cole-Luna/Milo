# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.3209 | 0.0000 | 0.0000 | 0.3209 | 0.3209 |
| MILO_LW | 0.3213 | 0.0000 | 0.0000 | 0.3213 | 0.3213 |
| SGD | 0.3012 | 0.0000 | 0.0000 | 0.3012 | 0.3012 |
| ADAMW | 0.5068 | 0.0000 | 0.0000 | 0.5068 | 0.5068 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.320854 | 0.321329 | MILO     |       nan |               | final_validation_loss |
| MILO          | SGD           | 0.320854 | 0.301197 | SGD      |       nan |               | final_validation_loss |
| MILO          | ADAMW         | 0.320854 | 0.506781 | MILO     |       nan |               | final_validation_loss |
| MILO_LW       | SGD           | 0.321329 | 0.301197 | SGD      |       nan |               | final_validation_loss |
| MILO_LW       | ADAMW         | 0.321329 | 0.506781 | MILO_LW  |       nan |               | final_validation_loss |
| SGD           | ADAMW         | 0.301197 | 0.506781 | SGD      |       nan |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 91.4198 | 0.0000 | 0.0000 | 91.4198 | 91.4198 |
| MILO_LW | 91.3827 | 0.0000 | 0.0000 | 91.3827 | 91.3827 |
| SGD | 91.7037 | 0.0000 | 0.0000 | 91.7037 | 91.7037 |
| ADAMW | 89.8148 | 0.0000 | 0.0000 | 89.8148 | 89.8148 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  91.4198 |  91.3827 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | SGD           |  91.4198 |  91.7037 | SGD      |       nan |               | final_validation_accuracy |
| MILO          | ADAMW         |  91.4198 |  89.8148 | MILO     |       nan |               | final_validation_accuracy |
| MILO_LW       | SGD           |  91.3827 |  91.7037 | SGD      |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  91.3827 |  89.8148 | MILO_LW  |       nan |               | final_validation_accuracy |
| SGD           | ADAMW         |  91.7037 |  89.8148 | SGD      |       nan |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9135 | 0.0000 | 0.0000 | 0.9135 | 0.9135 |
| MILO_LW | 0.9131 | 0.0000 | 0.0000 | 0.9131 | 0.9131 |
| SGD | 0.9163 | 0.0000 | 0.0000 | 0.9163 | 0.9163 |
| ADAMW | 0.8973 | 0.0000 | 0.0000 | 0.8973 | 0.8973 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.913457 | 0.913085 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | SGD           | 0.913457 | 0.916301 | SGD      |       nan |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.913457 | 0.897289 | MILO     |       nan |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.913085 | 0.916301 | SGD      |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.913085 | 0.897289 | MILO_LW  |       nan |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.916301 | 0.897289 | SGD      |       nan |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9922 | 0.0000 | 0.0000 | 0.9922 | 0.9922 |
| MILO_LW | 0.9921 | 0.0000 | 0.0000 | 0.9921 | 0.9921 |
| SGD | 0.9927 | 0.0000 | 0.0000 | 0.9927 | 0.9927 |
| ADAMW | 0.9907 | 0.0000 | 0.0000 | 0.9907 | 0.9907 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.99216  | 0.992052 | MILO     |       nan |               | final_validation_auc |
| MILO          | SGD           | 0.99216  | 0.992652 | SGD      |       nan |               | final_validation_auc |
| MILO          | ADAMW         | 0.99216  | 0.990691 | MILO     |       nan |               | final_validation_auc |
| MILO_LW       | SGD           | 0.992052 | 0.992652 | SGD      |       nan |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.992052 | 0.990691 | MILO_LW  |       nan |               | final_validation_auc |
| SGD           | ADAMW         | 0.992652 | 0.990691 | SGD      |       nan |               | final_validation_auc |

