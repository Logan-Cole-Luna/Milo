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
| milo_adaptive_True | 0.3787 | 0.0059 | 0.0034 | 0.3642 | 0.3933 |
| milo_adaptive_False | 2.3181 | 0.0010 | 0.0006 | 2.3157 | 2.3205 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better            |     p-value | Significant   | Metric     |
|:------------------|:-------------------|---------:|---------:|:------------------|------------:|:--------------|:-----------|
| milo_adaptive_True | milo_adaptive_False | 0.378722 |  2.31808 | milo_adaptive_True | 1.70246e-06 | ***           | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_adaptive_True | 89.6993 | 0.0775 | 0.0447 | 89.5069 | 89.8918 |
| milo_adaptive_False | 10.0567 | 0.2380 | 0.1374 | 9.4655 | 10.6478 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better            |     p-value | Significant   | Metric         |
|:------------------|:-------------------|---------:|---------:|:------------------|------------:|:--------------|:---------------|
| milo_adaptive_True | milo_adaptive_False |  89.6993 |  10.0567 | milo_adaptive_True | 3.15158e-07 | ***           | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_adaptive_True | 0.8970 | 0.0007 | 0.0004 | 0.8953 | 0.8988 |
| milo_adaptive_False | 0.0615 | 0.0025 | 0.0014 | 0.0553 | 0.0677 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A       | Optimizer B        |   Mean A |    Mean B | Better            |     p-value | Significant   | Metric         |
|:------------------|:-------------------|---------:|----------:|:------------------|------------:|:--------------|:---------------|
| milo_adaptive_True | milo_adaptive_False | 0.897047 | 0.0614907 | milo_adaptive_True | 4.77857e-07 | ***           | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

