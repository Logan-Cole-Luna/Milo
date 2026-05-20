# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.1045 | 0.0000 | 0.0000 | 0.1045 | 0.1045 |
| MILO_LW | 0.0959 | 0.0000 | 0.0000 | 0.0959 | 0.0959 |
| SGD | 0.1005 | 0.0000 | 0.0000 | 0.1005 | 0.1005 |
| ADAMW | 0.3184 | 0.0000 | 0.0000 | 0.3184 | 0.3184 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |    Mean B | Better   |   p-value | Significant   | Metric                |
|:--------------|:--------------|----------:|----------:|:---------|----------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.104499  | 0.0958847 | MILO_LW  |       nan |               | final_validation_loss |
| MILO          | SGD           | 0.104499  | 0.100478  | SGD      |       nan |               | final_validation_loss |
| MILO          | ADAMW         | 0.104499  | 0.318381  | MILO     |       nan |               | final_validation_loss |
| MILO_LW       | SGD           | 0.0958847 | 0.100478  | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADAMW         | 0.0958847 | 0.318381  | MILO_LW  |       nan |               | final_validation_loss |
| SGD           | ADAMW         | 0.100478  | 0.318381  | SGD      |       nan |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 97.4444 | 0.0000 | 0.0000 | 97.4444 | 97.4444 |
| MILO_LW | 97.4074 | 0.0000 | 0.0000 | 97.4074 | 97.4074 |
| SGD | 97.0617 | 0.0000 | 0.0000 | 97.0617 | 97.0617 |
| ADAMW | 92.1358 | 0.0000 | 0.0000 | 92.1358 | 92.1358 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  97.4444 |  97.4074 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | SGD           |  97.4444 |  97.0617 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAMW         |  97.4444 |  92.1358 | MILO     |       nan |               | final_validation_accuracy |
| MILO_LW       | SGD           |  97.4074 |  97.0617 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  97.4074 |  92.1358 | MILO_LW  |       nan |               | final_validation_accuracy |
| SGD           | ADAMW         |  97.0617 |  92.1358 | SGD      |       nan |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9743 | 0.0000 | 0.0000 | 0.9743 | 0.9743 |
| MILO_LW | 0.9740 | 0.0000 | 0.0000 | 0.9740 | 0.9740 |
| SGD | 0.9705 | 0.0000 | 0.0000 | 0.9705 | 0.9705 |
| ADAMW | 0.9210 | 0.0000 | 0.0000 | 0.9210 | 0.9210 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.97435  | 0.973999 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | SGD           | 0.97435  | 0.970471 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.97435  | 0.920971 | MILO     |       nan |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.973999 | 0.970471 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.973999 | 0.920971 | MILO_LW  |       nan |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.970471 | 0.920971 | SGD      |       nan |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9994 | 0.0000 | 0.0000 | 0.9994 | 0.9994 |
| MILO_LW | 0.9995 | 0.0000 | 0.0000 | 0.9995 | 0.9995 |
| SGD | 0.9992 | 0.0000 | 0.0000 | 0.9992 | 0.9992 |
| ADAMW | 0.9939 | 0.0000 | 0.0000 | 0.9939 | 0.9939 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.99943  | 0.999517 | MILO_LW  |       nan |               | final_validation_auc |
| MILO          | SGD           | 0.99943  | 0.999176 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADAMW         | 0.99943  | 0.993877 | MILO     |       nan |               | final_validation_auc |
| MILO_LW       | SGD           | 0.999517 | 0.999176 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.999517 | 0.993877 | MILO_LW  |       nan |               | final_validation_auc |
| SGD           | ADAMW         | 0.999176 | 0.993877 | SGD      |       nan |               | final_validation_auc |

