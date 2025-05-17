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
| milo_normalize_True | 0.3795 | 0.0232 | 0.0134 | 0.3218 | 0.4373 |
| milo_normalize_False | 0.3454 | 0.0103 | 0.0059 | 0.3199 | 0.3709 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better              |   p-value | Significant   | Metric     |
|:-------------------|:--------------------|---------:|---------:|:--------------------|----------:|:--------------|:-----------|
| milo_normalize_True | milo_normalize_False | 0.379532 | 0.345406 | milo_normalize_False |  0.110311 |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_normalize_True | 89.8427 | 0.4361 | 0.2518 | 88.7593 | 90.9260 |
| milo_normalize_False | 89.2407 | 0.0583 | 0.0337 | 89.0959 | 89.3855 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric         |
|:-------------------|:--------------------|---------:|---------:|:-------------------|----------:|:--------------|:---------------|
| milo_normalize_True | milo_normalize_False |  89.8427 |  89.2407 | milo_normalize_True |  0.136935 |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_normalize_True | 0.8984 | 0.0044 | 0.0025 | 0.8876 | 0.9093 |
| milo_normalize_False | 0.8925 | 0.0006 | 0.0004 | 0.8910 | 0.8941 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric         |
|:-------------------|:--------------------|---------:|---------:|:-------------------|----------:|:--------------|:---------------|
| milo_normalize_True | milo_normalize_False | 0.898436 | 0.892547 | milo_normalize_True |  0.141971 |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

