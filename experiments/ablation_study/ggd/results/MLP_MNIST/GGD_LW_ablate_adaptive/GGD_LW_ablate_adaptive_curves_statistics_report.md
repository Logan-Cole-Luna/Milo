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
| milo_LW_baseline_adaptive | 0.0024 | 0.0020 | 0.0012 | -0.0026 | 0.0075 |
| milo_LW_adaptive_False | 2.5808 | 0.0875 | 0.0505 | 2.3635 | 2.7982 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A              | Optimizer B           |     Mean A |   Mean B | Better                   |     p-value | Significant   | Metric     |
|:-------------------------|:----------------------|-----------:|---------:|:-------------------------|------------:|:--------------|:-----------|
| milo_LW_baseline_adaptive | milo_LW_adaptive_False | 0.00241545 |  2.58083 | milo_LW_baseline_adaptive | 0.000381103 | ***           | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_baseline_adaptive | 99.9483 | 0.0536 | 0.0310 | 99.8151 | 100.0816 |
| milo_LW_adaptive_False | 19.4239 | 5.5757 | 3.2191 | 5.5730 | 33.2748 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A              | Optimizer B           |   Mean A |   Mean B | Better                   |    p-value | Significant   | Metric         |
|:-------------------------|:----------------------|---------:|---------:|:-------------------------|-----------:|:--------------|:---------------|
| milo_LW_baseline_adaptive | milo_LW_adaptive_False |  99.9483 |  19.4239 | milo_LW_baseline_adaptive | 0.00159299 | **            | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_baseline_adaptive | 0.9995 | 0.0005 | 0.0003 | 0.9981 | 1.0008 |
| milo_LW_adaptive_False | 0.2011 | 0.0647 | 0.0373 | 0.0405 | 0.3617 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A              | Optimizer B           |   Mean A |   Mean B | Better                   |    p-value | Significant   | Metric         |
|:-------------------------|:----------------------|---------:|---------:|:-------------------------|-----------:|:--------------|:---------------|
| milo_LW_baseline_adaptive | milo_LW_adaptive_False | 0.999483 | 0.201113 | milo_LW_baseline_adaptive | 0.00217757 | **            | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

