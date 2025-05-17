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
| milo_weight_decay_0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_weight_decay_0.001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_weight_decay_0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A             | Optimizer B            |      Mean A |      Mean B | Better                 |   p-value | Significant   | Metric     |
|:------------------------|:-----------------------|------------:|------------:|:-----------------------|----------:|:--------------|:-----------|
| milo_weight_decay_0.0001 | milo_weight_decay_0.001 | 2.90912e-07 | 1.97287e-07 | milo_weight_decay_0.001 | 0.268981  |               | final_loss |
| milo_weight_decay_0.0001 | milo_weight_decay_0.0   | 2.90912e-07 | 9.42813e-08 | milo_weight_decay_0.0   | 0.074637  |               | final_loss |
| milo_weight_decay_0.001  | milo_weight_decay_0.0   | 1.97287e-07 | 9.42813e-08 | milo_weight_decay_0.0   | 0.0214753 | *             | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_weight_decay_0.0001 | 100.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 |
| milo_weight_decay_0.001 | 100.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 |
| milo_weight_decay_0.0 | 100.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A             | Optimizer B            |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric         |
|:------------------------|:-----------------------|---------:|---------:|:-----------------------|----------:|:--------------|:---------------|
| milo_weight_decay_0.0001 | milo_weight_decay_0.001 |      100 |      100 | milo_weight_decay_0.001 |       nan |               | final_accuracy |
| milo_weight_decay_0.0001 | milo_weight_decay_0.0   |      100 |      100 | milo_weight_decay_0.0   |       nan |               | final_accuracy |
| milo_weight_decay_0.001  | milo_weight_decay_0.0   |      100 |      100 | milo_weight_decay_0.0   |       nan |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_weight_decay_0.0001 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| milo_weight_decay_0.001 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| milo_weight_decay_0.0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A             | Optimizer B            |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric         |
|:------------------------|:-----------------------|---------:|---------:|:-----------------------|----------:|:--------------|:---------------|
| milo_weight_decay_0.0001 | milo_weight_decay_0.001 |        1 |        1 | milo_weight_decay_0.001 |       nan |               | final_f1_score |
| milo_weight_decay_0.0001 | milo_weight_decay_0.0   |        1 |        1 | milo_weight_decay_0.0   |       nan |               | final_f1_score |
| milo_weight_decay_0.001  | milo_weight_decay_0.0   |        1 |        1 | milo_weight_decay_0.0   |       nan |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

