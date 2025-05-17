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
| milo_weight_decay_0.0001 | 0.3845 | 0.0185 | 0.0107 | 0.3385 | 0.4306 |
| milo_weight_decay_0.001 | 0.3406 | 0.0088 | 0.0051 | 0.3188 | 0.3625 |
| milo_weight_decay_0.0 | 0.3417 | 0.0075 | 0.0043 | 0.3230 | 0.3604 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A             | Optimizer B            |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric     |
|:------------------------|:-----------------------|---------:|---------:|:-----------------------|----------:|:--------------|:-----------|
| milo_weight_decay_0.0001 | milo_weight_decay_0.001 | 0.384544 | 0.340636 | milo_weight_decay_0.001 | 0.0369335 | *             | final_loss |
| milo_weight_decay_0.0001 | milo_weight_decay_0.0   | 0.384544 | 0.341741 | milo_weight_decay_0.0   | 0.0421252 | *             | final_loss |
| milo_weight_decay_0.001  | milo_weight_decay_0.0   | 0.340636 | 0.341741 | milo_weight_decay_0.001 | 0.877048  |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_weight_decay_0.0001 | 88.5593 | 0.2961 | 0.1710 | 87.8237 | 89.2949 |
| milo_weight_decay_0.001 | 89.4933 | 0.3177 | 0.1834 | 88.7040 | 90.2827 |
| milo_weight_decay_0.0 | 89.9227 | 0.1652 | 0.0954 | 89.5124 | 90.3330 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A             | Optimizer B            |   Mean A |   Mean B | Better                 |    p-value | Significant   | Metric         |
|:------------------------|:-----------------------|---------:|---------:|:-----------------------|-----------:|:--------------|:---------------|
| milo_weight_decay_0.0001 | milo_weight_decay_0.001 |  88.5593 |  89.4933 | milo_weight_decay_0.001 | 0.0205729  | *             | final_accuracy |
| milo_weight_decay_0.0001 | milo_weight_decay_0.0   |  88.5593 |  89.9227 | milo_weight_decay_0.0   | 0.00525958 | **            | final_accuracy |
| milo_weight_decay_0.001  | milo_weight_decay_0.0   |  89.4933 |  89.9227 | milo_weight_decay_0.0   | 0.12921    |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_weight_decay_0.0001 | 0.8856 | 0.0030 | 0.0018 | 0.8781 | 0.8931 |
| milo_weight_decay_0.001 | 0.8949 | 0.0031 | 0.0018 | 0.8872 | 0.9026 |
| milo_weight_decay_0.0 | 0.8991 | 0.0017 | 0.0010 | 0.8950 | 0.9033 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A             | Optimizer B            |   Mean A |   Mean B | Better                 |    p-value | Significant   | Metric         |
|:------------------------|:-----------------------|---------:|---------:|:-----------------------|-----------:|:--------------|:---------------|
| milo_weight_decay_0.0001 | milo_weight_decay_0.001 | 0.885585 | 0.894892 | milo_weight_decay_0.001 | 0.0205673  | *             | final_f1_score |
| milo_weight_decay_0.0001 | milo_weight_decay_0.0   | 0.885585 | 0.899135 | milo_weight_decay_0.0   | 0.00583629 | **            | final_f1_score |
| milo_weight_decay_0.001  | milo_weight_decay_0.0   | 0.894892 | 0.899135 | milo_weight_decay_0.0   | 0.126357   |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

