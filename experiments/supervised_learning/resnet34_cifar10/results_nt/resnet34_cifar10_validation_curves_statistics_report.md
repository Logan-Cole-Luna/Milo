# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.7639 | 0.0000 | 0.0000 | 0.7639 | 0.7639 |
| MILO_LW | 0.7482 | 0.0000 | 0.0000 | 0.7482 | 0.7482 |
| SGD | 0.8808 | 0.0000 | 0.0000 | 0.8808 | 0.8808 |
| ADAMW | 0.6949 | 0.0000 | 0.0000 | 0.6949 | 0.6949 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.763945 | 0.748202 | MILO_LW  |       nan |               | final_validation_loss |
| MILO          | SGD           | 0.763945 | 0.880751 | MILO     |       nan |               | final_validation_loss |
| MILO          | ADAMW         | 0.763945 | 0.69489  | ADAMW    |       nan |               | final_validation_loss |
| MILO_LW       | SGD           | 0.748202 | 0.880751 | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADAMW         | 0.748202 | 0.69489  | ADAMW    |       nan |               | final_validation_loss |
| SGD           | ADAMW         | 0.880751 | 0.69489  | ADAMW    |       nan |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 74.3111 | 0.0000 | 0.0000 | 74.3111 | 74.3111 |
| MILO_LW | 75.7333 | 0.0000 | 0.0000 | 75.7333 | 75.7333 |
| SGD | 71.5259 | 0.0000 | 0.0000 | 71.5259 | 71.5259 |
| ADAMW | 77.2741 | 0.0000 | 0.0000 | 77.2741 | 77.2741 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  74.3111 |  75.7333 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO          | SGD           |  74.3111 |  71.5259 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAMW         |  74.3111 |  77.2741 | ADAMW    |       nan |               | final_validation_accuracy |
| MILO_LW       | SGD           |  75.7333 |  71.5259 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  75.7333 |  77.2741 | ADAMW    |       nan |               | final_validation_accuracy |
| SGD           | ADAMW         |  71.5259 |  77.2741 | ADAMW    |       nan |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.7412 | 0.0000 | 0.0000 | 0.7412 | 0.7412 |
| MILO_LW | 0.7571 | 0.0000 | 0.0000 | 0.7571 | 0.7571 |
| SGD | 0.7110 | 0.0000 | 0.0000 | 0.7110 | 0.7110 |
| ADAMW | 0.7720 | 0.0000 | 0.0000 | 0.7720 | 0.7720 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.741194 | 0.75708  | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO          | SGD           | 0.741194 | 0.711027 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.741194 | 0.77201  | ADAMW    |       nan |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.75708  | 0.711027 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.75708  | 0.77201  | ADAMW    |       nan |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.711027 | 0.77201  | ADAMW    |       nan |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9675 | 0.0000 | 0.0000 | 0.9675 | 0.9675 |
| MILO_LW | 0.9692 | 0.0000 | 0.0000 | 0.9692 | 0.9692 |
| SGD | 0.9580 | 0.0000 | 0.0000 | 0.9580 | 0.9580 |
| ADAMW | 0.9740 | 0.0000 | 0.0000 | 0.9740 | 0.9740 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.967544 | 0.969192 | MILO_LW  |       nan |               | final_validation_auc |
| MILO          | SGD           | 0.967544 | 0.957968 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADAMW         | 0.967544 | 0.974013 | ADAMW    |       nan |               | final_validation_auc |
| MILO_LW       | SGD           | 0.969192 | 0.957968 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.969192 | 0.974013 | ADAMW    |       nan |               | final_validation_auc |
| SGD           | ADAMW         | 0.957968 | 0.974013 | ADAMW    |       nan |               | final_validation_auc |

