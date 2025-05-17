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
| milo_baseline_adaptive | 0.0004 | 0.0006 | 0.0004 | -0.0012 | 0.0020 |
| milo_adaptive_False | 0.4067 | 0.1714 | 0.0989 | -0.0191 | 0.8324 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A           | Optimizer B        |      Mean A |   Mean B | Better                |   p-value | Significant   | Metric     |
|:----------------------|:-------------------|------------:|---------:|:----------------------|----------:|:--------------|:-----------|
| milo_baseline_adaptive | milo_adaptive_False | 0.000376966 | 0.406666 | milo_baseline_adaptive | 0.0545058 |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_baseline_adaptive | 99.9894 | 0.0183 | 0.0106 | 99.9440 | 100.0349 |
| milo_adaptive_False | 96.0811 | 1.2104 | 0.6989 | 93.0742 | 99.0880 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A           | Optimizer B        |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric         |
|:----------------------|:-------------------|---------:|---------:|:----------------------|----------:|:--------------|:---------------|
| milo_baseline_adaptive | milo_adaptive_False |  99.9894 |  96.0811 | milo_baseline_adaptive | 0.0304928 | *             | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_baseline_adaptive | 0.9999 | 0.0002 | 0.0001 | 0.9994 | 1.0003 |
| milo_adaptive_False | 0.9621 | 0.0114 | 0.0066 | 0.9336 | 0.9905 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A           | Optimizer B        |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric         |
|:----------------------|:-------------------|---------:|---------:|:----------------------|----------:|:--------------|:---------------|
| milo_baseline_adaptive | milo_adaptive_False | 0.999895 | 0.962077 | milo_baseline_adaptive | 0.0291927 | *             | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

