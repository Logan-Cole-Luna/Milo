# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_weight_decay_0.0001 | 0.1326 | 0.0018 | 0.0013 | 0.1165 | 0.1487 |
| MILO_LW_weight_decay_0.001 | 0.1285 | 0.0026 | 0.0018 | 0.1051 | 0.1520 |
| MILO_LW_weight_decay_0.0 | 0.1366 | 0.0058 | 0.0041 | 0.0842 | 0.1889 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A                 | Optimizer B                |   Mean A |   Mean B | Better                      |   p-value | Significant   | Metric                |
|:----------------------------|:---------------------------|---------:|---------:|:----------------------------|----------:|:--------------|:----------------------|
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.001 | 0.132584 | 0.128548 | MILO_LW_weight_decay_0.001  |  0.228739 |               | final_validation_loss |
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.0   | 0.132584 | 0.136553 | MILO_LW_weight_decay_0.0001 |  0.506078 |               | final_validation_loss |
| MILO_LW_weight_decay_0.001  | MILO_LW_weight_decay_0.0   | 0.128548 | 0.136553 | MILO_LW_weight_decay_0.001  |  0.270682 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_weight_decay_0.0001 | 96.0350 | 0.1673 | 0.1183 | 94.5314 | 97.5386 |
| MILO_LW_weight_decay_0.001 | 95.8958 | 0.0295 | 0.0208 | 95.6311 | 96.1605 |
| MILO_LW_weight_decay_0.0 | 95.6033 | 0.1838 | 0.1300 | 93.9515 | 97.2551 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A                 | Optimizer B                |   Mean A |   Mean B | Better                      |   p-value | Significant   | Metric                    |
|:----------------------------|:---------------------------|---------:|---------:|:----------------------------|----------:|:--------------|:--------------------------|
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.001 |  96.035  |  95.8958 | MILO_LW_weight_decay_0.0001 |  0.444595 |               | final_validation_accuracy |
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.0   |  96.035  |  95.6033 | MILO_LW_weight_decay_0.0001 |  0.134533 |               | final_validation_accuracy |
| MILO_LW_weight_decay_0.001  | MILO_LW_weight_decay_0.0   |  95.8958 |  95.6033 | MILO_LW_weight_decay_0.001  |  0.259471 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_weight_decay_0.0001 | 0.9624 | 0.0015 | 0.0010 | 0.9490 | 0.9757 |
| MILO_LW_weight_decay_0.001 | 0.9611 | 0.0003 | 0.0002 | 0.9588 | 0.9634 |
| MILO_LW_weight_decay_0.0 | 0.9586 | 0.0016 | 0.0011 | 0.9443 | 0.9729 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A                 | Optimizer B                |   Mean A |   Mean B | Better                      |   p-value | Significant   | Metric                    |
|:----------------------------|:---------------------------|---------:|---------:|:----------------------------|----------:|:--------------|:--------------------------|
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.001 | 0.96235  | 0.961134 | MILO_LW_weight_decay_0.0001 |  0.449511 |               | final_validation_f1_score |
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.0   | 0.96235  | 0.958566 | MILO_LW_weight_decay_0.0001 |  0.133784 |               | final_validation_f1_score |
| MILO_LW_weight_decay_0.001  | MILO_LW_weight_decay_0.0   | 0.961134 | 0.958566 | MILO_LW_weight_decay_0.001  |  0.256109 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_weight_decay_0.0001 | 0.9987 | 0.0001 | 0.0001 | 0.9978 | 0.9995 |
| MILO_LW_weight_decay_0.001 | 0.9986 | 0.0000 | 0.0000 | 0.9984 | 0.9988 |
| MILO_LW_weight_decay_0.0 | 0.9984 | 0.0001 | 0.0001 | 0.9974 | 0.9995 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A                 | Optimizer B                |   Mean A |   Mean B | Better                      |   p-value | Significant   | Metric               |
|:----------------------------|:---------------------------|---------:|---------:|:----------------------------|----------:|:--------------|:---------------------|
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.001 | 0.998653 | 0.998598 | MILO_LW_weight_decay_0.0001 |  0.556434 |               | final_validation_auc |
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.0   | 0.998653 | 0.998434 | MILO_LW_weight_decay_0.0001 |  0.175844 |               | final_validation_auc |
| MILO_LW_weight_decay_0.001  | MILO_LW_weight_decay_0.0   | 0.998598 | 0.998434 | MILO_LW_weight_decay_0.001  |  0.281689 |               | final_validation_auc |

