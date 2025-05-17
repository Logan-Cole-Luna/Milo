# Statistical Analysis Report

## Statistical Methodology

This report documents the statistical methods used to analyze the experimental results.

### Error Calculation

- **Standard Deviation**: Calculated using the unbiased estimator (n-1 denominator)
- **Standard Error of the Mean (SEM)**: SD/√n where n is the number of runs
- **95% Confidence Intervals**: Using t-distribution (t-critical × SEM)
- **Error Bands in Plots**: Represent ±1 SEM around the mean

### Statistical Assumptions

- Errors are assumed to follow a t-distribution due to small sample sizes
- Runs are treated as independent samples with the same underlying distribution
- Variances between different optimizers may be unequal (Welch's t-test used)

### Sources of Variability

- Random initialization of model parameters
- Stochastic gradient descent optimization
- Batch sampling during training

## Statistical Results

### Loss Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_weight_decay_0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_LW_weight_decay_0.001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_LW_weight_decay_0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A                | Optimizer B               |      Mean A |      Mean B | Better                     |    p-value | Significant   | Metric     |
|:---------------------------|:--------------------------|------------:|------------:|:---------------------------|-----------:|:--------------|:-----------|
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.001 | 2.4976e-07  | 4.11153e-07 | milo_LW_weight_decay_0.0001 | 0.00319364 | **            | final_loss |
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.0   | 2.4976e-07  | 2.81463e-07 | milo_LW_weight_decay_0.0001 | 0.328236   |               | final_loss |
| milo_LW_weight_decay_0.001  | milo_LW_weight_decay_0.0   | 4.11153e-07 | 2.81463e-07 | milo_LW_weight_decay_0.0    | 0.0174561  | *             | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_weight_decay_0.0001 | 100.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 |
| milo_LW_weight_decay_0.001 | 100.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 |
| milo_LW_weight_decay_0.0 | 100.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A                | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric         |
|:---------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:---------------|
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.001 |      100 |      100 | milo_LW_weight_decay_0.001 |       nan |               | final_accuracy |
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.0   |      100 |      100 | milo_LW_weight_decay_0.0   |       nan |               | final_accuracy |
| milo_LW_weight_decay_0.001  | milo_LW_weight_decay_0.0   |      100 |      100 | milo_LW_weight_decay_0.0   |       nan |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_weight_decay_0.0001 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| milo_LW_weight_decay_0.001 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| milo_LW_weight_decay_0.0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A                | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric         |
|:---------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:---------------|
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.001 |        1 |        1 | milo_LW_weight_decay_0.001 |       nan |               | final_f1_score |
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.0   |        1 |        1 | milo_LW_weight_decay_0.0   |       nan |               | final_f1_score |
| milo_LW_weight_decay_0.001  | milo_LW_weight_decay_0.0   |        1 |        1 | milo_LW_weight_decay_0.0   |       nan |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

