# Statistical Analysis Report

## Methodology

Standard deviations, standard errors, and 95% confidence intervals were computed using the t-distribution.

### Validation_Loss Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_momentum_0.9 | 0.4241 | 0.0024 | 0.0014 | 0.4182 | 0.4300 |
| milo_momentum_0.0 | 0.4557 | 0.0227 | 0.0131 | 0.3993 | 0.5122 |
| milo_momentum_0.45 | 0.4481 | 0.0154 | 0.0089 | 0.4097 | 0.4865 |

#### Pairwise Significance Tests

| Optimizer A      | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric                |
|:-----------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:----------------------|
| milo_momentum_0.9 | milo_momentum_0.0  | 0.424053 | 0.455736 | milo_momentum_0.9  |  0.135583 |               | final_validation_loss |
| milo_momentum_0.9 | milo_momentum_0.45 | 0.424053 | 0.448084 | milo_momentum_0.9  |  0.11133  |               | final_validation_loss |
| milo_momentum_0.0 | milo_momentum_0.45 | 0.455736 | 0.448084 | milo_momentum_0.45 |  0.657942 |               | final_validation_loss |

### Validation_Accuracy Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_momentum_0.9 | 85.8067 | 0.3395 | 0.1960 | 84.9634 | 86.6499 |
| milo_momentum_0.0 | 84.7833 | 0.9005 | 0.5199 | 82.5465 | 87.0202 |
| milo_momentum_0.45 | 84.8933 | 0.6204 | 0.3582 | 83.3521 | 86.4346 |

#### Pairwise Significance Tests

| Optimizer A      | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric                    |
|:-----------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:--------------------------|
| milo_momentum_0.9 | milo_momentum_0.0  |  85.8067 |  84.7833 | milo_momentum_0.9  |  0.178443 |               | final_validation_accuracy |
| milo_momentum_0.9 | milo_momentum_0.45 |  85.8067 |  84.8933 | milo_momentum_0.9  |  0.108446 |               | final_validation_accuracy |
| milo_momentum_0.0 | milo_momentum_0.45 |  84.7833 |  84.8933 | milo_momentum_0.45 |  0.871159 |               | final_validation_accuracy |

### Validation_F1_Score Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_momentum_0.9 | 0.8578 | 0.0039 | 0.0022 | 0.8481 | 0.8674 |
| milo_momentum_0.0 | 0.8476 | 0.0087 | 0.0050 | 0.8259 | 0.8692 |
| milo_momentum_0.45 | 0.8481 | 0.0058 | 0.0033 | 0.8338 | 0.8624 |

#### Pairwise Significance Tests

| Optimizer A      | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric                    |
|:-----------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:--------------------------|
| milo_momentum_0.9 | milo_momentum_0.0  | 0.857797 | 0.847584 | milo_momentum_0.9  | 0.168475  |               | final_validation_f1_score |
| milo_momentum_0.9 | milo_momentum_0.45 | 0.857797 | 0.848071 | milo_momentum_0.9  | 0.0814198 |               | final_validation_f1_score |
| milo_momentum_0.0 | milo_momentum_0.45 | 0.847584 | 0.848071 | milo_momentum_0.45 | 0.939973  |               | final_validation_f1_score |

### Validation_Auc Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_momentum_0.9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_momentum_0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_momentum_0.45 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Pairwise Significance Tests

| Optimizer A      | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric               |
|:-----------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:---------------------|
| milo_momentum_0.9 | milo_momentum_0.0  |        0 |        0 | milo_momentum_0.0  |       nan |               | final_validation_auc |
| milo_momentum_0.9 | milo_momentum_0.45 |        0 |        0 | milo_momentum_0.45 |       nan |               | final_validation_auc |
| milo_momentum_0.0 | milo_momentum_0.45 |        0 |        0 | milo_momentum_0.45 |       nan |               | final_validation_auc |

