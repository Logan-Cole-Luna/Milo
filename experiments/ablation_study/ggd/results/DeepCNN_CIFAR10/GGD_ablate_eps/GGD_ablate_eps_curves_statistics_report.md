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
| milo_eps_1e-05 | 0.3402 | 0.0175 | 0.0101 | 0.2968 | 0.3836 |
| milo_eps_0.005 | 0.2845 | 0.0110 | 0.0064 | 0.2572 | 0.3119 |
| milo_eps_0.05 | 0.2410 | 0.0044 | 0.0025 | 0.2301 | 0.2520 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |   p-value | Significant   | Metric     |
|:--------------|:--------------|---------:|---------:|:--------------|----------:|:--------------|:-----------|
| milo_eps_1e-05 | milo_eps_0.005 | 0.340209 | 0.284547 | milo_eps_0.005 | 0.0142493 | *             | final_loss |
| milo_eps_1e-05 | milo_eps_0.05  | 0.340209 | 0.241044 | milo_eps_0.05  | 0.0072666 | **            | final_loss |
| milo_eps_0.005 | milo_eps_0.05  | 0.284547 | 0.241044 | milo_eps_0.05  | 0.0116879 | *             | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_eps_1e-05 | 90.1907 | 0.2718 | 0.1569 | 89.5155 | 90.8659 |
| milo_eps_0.005 | 91.1173 | 0.3049 | 0.1760 | 90.3599 | 91.8748 |
| milo_eps_0.05 | 92.2020 | 0.2007 | 0.1159 | 91.7033 | 92.7007 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |     p-value | Significant   | Metric         |
|:--------------|:--------------|---------:|---------:|:--------------|------------:|:--------------|:---------------|
| milo_eps_1e-05 | milo_eps_0.005 |  90.1907 |  91.1173 | milo_eps_0.005 | 0.0175446   | *             | final_accuracy |
| milo_eps_1e-05 | milo_eps_0.05  |  90.1907 |  92.202  | milo_eps_0.05  | 0.000756623 | ***           | final_accuracy |
| milo_eps_0.005 | milo_eps_0.05  |  91.1173 |  92.202  | milo_eps_0.05  | 0.00992648  | **            | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_eps_1e-05 | 0.9018 | 0.0027 | 0.0016 | 0.8951 | 0.9085 |
| milo_eps_0.005 | 0.9112 | 0.0031 | 0.0018 | 0.9035 | 0.9189 |
| milo_eps_0.05 | 0.9221 | 0.0020 | 0.0012 | 0.9171 | 0.9271 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |     p-value | Significant   | Metric         |
|:--------------|:--------------|---------:|---------:|:--------------|------------:|:--------------|:---------------|
| milo_eps_1e-05 | milo_eps_0.005 | 0.901807 | 0.911188 | milo_eps_0.005 | 0.0176309   | *             | final_f1_score |
| milo_eps_1e-05 | milo_eps_0.05  | 0.901807 | 0.922081 | milo_eps_0.05  | 0.000726929 | ***           | final_f1_score |
| milo_eps_0.005 | milo_eps_0.05  | 0.911188 | 0.922081 | milo_eps_0.05  | 0.0105562   | *             | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

