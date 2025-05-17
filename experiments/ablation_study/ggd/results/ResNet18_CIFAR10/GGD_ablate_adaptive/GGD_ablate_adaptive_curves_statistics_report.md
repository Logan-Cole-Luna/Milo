# Statistical Analysis Report

## Methodology

Standard deviations, standard errors, and 95% confidence intervals were computed using the t-distribution.

### Validation_Loss Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_adaptive_True | 0.4342 | 0.0351 | 0.0203 | 0.3470 | 0.5215 |
| milo_adaptive_False | 675.1288 | 380.6141 | 219.7477 | -270.3691 | 1620.6268 |

#### Pairwise Significance Tests

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric                |
|:------------------|:-------------------|---------:|---------:|:------------------|----------:|:--------------|:----------------------|
| milo_adaptive_True | milo_adaptive_False | 0.434237 |  675.129 | milo_adaptive_True | 0.0917195 |               | final_validation_loss |

### Validation_Accuracy Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_adaptive_True | 85.5633 | 1.3299 | 0.7678 | 82.2598 | 88.8669 |
| milo_adaptive_False | 49.1733 | 10.1113 | 5.8378 | 24.0555 | 74.2912 |

#### Pairwise Significance Tests

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric                    |
|:------------------|:-------------------|---------:|---------:|:------------------|----------:|:--------------|:--------------------------|
| milo_adaptive_True | milo_adaptive_False |  85.5633 |  49.1733 | milo_adaptive_True | 0.0231902 | *             | final_validation_accuracy |

### Validation_F1_Score Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_adaptive_True | 0.8555 | 0.0131 | 0.0076 | 0.8229 | 0.8880 |
| milo_adaptive_False | 0.4736 | 0.1030 | 0.0595 | 0.2178 | 0.7294 |

#### Pairwise Significance Tests

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric                    |
|:------------------|:-------------------|---------:|---------:|:------------------|----------:|:--------------|:--------------------------|
| milo_adaptive_True | milo_adaptive_False | 0.855487 | 0.473594 | milo_adaptive_True | 0.0219379 | *             | final_validation_f1_score |

### Validation_Auc Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_adaptive_True | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_adaptive_False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Pairwise Significance Tests

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric               |
|:------------------|:-------------------|---------:|---------:|:-------------------|----------:|:--------------|:---------------------|
| milo_adaptive_True | milo_adaptive_False |        0 |        0 | milo_adaptive_False |       nan |               | final_validation_auc |

