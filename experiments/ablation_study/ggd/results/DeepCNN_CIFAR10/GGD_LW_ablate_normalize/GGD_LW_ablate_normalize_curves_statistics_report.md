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
| milo_LW_normalize_True | 0.1334 | 0.0058 | 0.0034 | 0.1189 | 0.1479 |
| milo_LW_normalize_False | 0.1610 | 0.0104 | 0.0060 | 0.1352 | 0.1869 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric     |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|----------:|:--------------|:-----------|
| milo_LW_normalize_True | milo_LW_normalize_False | 0.133401 | 0.161031 | milo_LW_normalize_True | 0.0254101 | *             | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_normalize_True | 95.3293 | 0.1599 | 0.0923 | 94.9322 | 95.7265 |
| milo_LW_normalize_False | 94.3667 | 0.3571 | 0.2062 | 93.4795 | 95.2538 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric         |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|----------:|:--------------|:---------------|
| milo_LW_normalize_True | milo_LW_normalize_False |  95.3293 |  94.3667 | milo_LW_normalize_True |   0.02775 | *             | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_normalize_True | 0.9533 | 0.0016 | 0.0009 | 0.9493 | 0.9573 |
| milo_LW_normalize_False | 0.9437 | 0.0036 | 0.0021 | 0.9348 | 0.9526 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric         |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|----------:|:--------------|:---------------|
| milo_LW_normalize_True | milo_LW_normalize_False | 0.953299 | 0.943667 | milo_LW_normalize_True | 0.0278795 | *             | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

