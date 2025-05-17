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
| milo_LW_scale_factor_0.2 | 0.9983 | 0.1691 | 0.0976 | 0.5782 | 1.4184 |
| milo_LW_scale_factor_0.1 | 0.6597 | 0.0965 | 0.0557 | 0.4201 | 0.8994 |
| milo_LW_scale_factor_0.5 | 0.4385 | 0.0469 | 0.0271 | 0.3220 | 0.5551 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A             | Optimizer B             |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric     |
|:------------------------|:------------------------|---------:|---------:|:------------------------|----------:|:--------------|:-----------|
| milo_LW_scale_factor_0.2 | milo_LW_scale_factor_0.1 | 0.998288 | 0.659747 | milo_LW_scale_factor_0.1 | 0.0530889 |               | final_loss |
| milo_LW_scale_factor_0.2 | milo_LW_scale_factor_0.5 | 0.998288 | 0.438545 | milo_LW_scale_factor_0.5 | 0.02256   | *             | final_loss |
| milo_LW_scale_factor_0.1 | milo_LW_scale_factor_0.5 | 0.659747 | 0.438545 | milo_LW_scale_factor_0.5 | 0.0397055 | *             | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_scale_factor_0.2 | 64.4707 | 6.5240 | 3.7666 | 48.2641 | 80.6772 |
| milo_LW_scale_factor_0.1 | 77.5067 | 4.0516 | 2.3392 | 67.4420 | 87.5713 |
| milo_LW_scale_factor_0.5 | 85.8167 | 1.5375 | 0.8877 | 81.9972 | 89.6361 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A             | Optimizer B             |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric         |
|:------------------------|:------------------------|---------:|---------:|:------------------------|----------:|:--------------|:---------------|
| milo_LW_scale_factor_0.2 | milo_LW_scale_factor_0.1 |  64.4707 |  77.5067 | milo_LW_scale_factor_0.1 | 0.0529264 |               | final_accuracy |
| milo_LW_scale_factor_0.2 | milo_LW_scale_factor_0.5 |  64.4707 |  85.8167 | milo_LW_scale_factor_0.5 | 0.0246967 | *             | final_accuracy |
| milo_LW_scale_factor_0.1 | milo_LW_scale_factor_0.5 |  77.5067 |  85.8167 | milo_LW_scale_factor_0.5 | 0.0564998 |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_scale_factor_0.2 | 0.6401 | 0.0695 | 0.0402 | 0.4674 | 0.8129 |
| milo_LW_scale_factor_0.1 | 0.7754 | 0.0404 | 0.0233 | 0.6750 | 0.8758 |
| milo_LW_scale_factor_0.5 | 0.8587 | 0.0155 | 0.0090 | 0.8201 | 0.8973 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A             | Optimizer B             |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric         |
|:------------------------|:------------------------|---------:|---------:|:------------------------|----------:|:--------------|:---------------|
| milo_LW_scale_factor_0.2 | milo_LW_scale_factor_0.1 | 0.640138 | 0.775391 | milo_LW_scale_factor_0.1 | 0.056929  |               | final_f1_score |
| milo_LW_scale_factor_0.2 | milo_LW_scale_factor_0.5 | 0.640138 | 0.858686 | milo_LW_scale_factor_0.5 | 0.0273463 | *             | final_f1_score |
| milo_LW_scale_factor_0.1 | milo_LW_scale_factor_0.5 | 0.775391 | 0.858686 | milo_LW_scale_factor_0.5 | 0.0556954 |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

