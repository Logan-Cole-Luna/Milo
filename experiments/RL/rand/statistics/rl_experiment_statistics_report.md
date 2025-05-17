# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Success Rate Statistics

Number of runs: 5

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |    p-value | Significant   | Metric       |
|:--------------|:--------------|---------:|---------:|:---------|-----------:|:--------------|:-------------|
| MILO          | MILO_LW       |      0.2 |      0.4 | MILO_LW  |   0.545424 |               | success_rate |
| MILO          | SGD           |      0.2 |      0   | MILO     |   0.373901 |               | success_rate |
| MILO          | ADAMW         |      0.2 |      0   | MILO     |   0.373901 |               | success_rate |
| MILO          | ADAGRAD       |      0.2 |      0.2 | ADAGRAD  |   1        |               | success_rate |
| MILO          | NOVOGRAD      |      0.2 |      0   | MILO     |   0.373901 |               | success_rate |
| MILO_LW       | SGD           |      0.4 |      0   | MILO_LW  |   0.177808 |               | success_rate |
| MILO_LW       | ADAMW         |      0.4 |      0   | MILO_LW  |   0.177808 |               | success_rate |
| MILO_LW       | ADAGRAD       |      0.4 |      0.2 | MILO_LW  |   0.545424 |               | success_rate |
| MILO_LW       | NOVOGRAD      |      0.4 |      0   | MILO_LW  |   0.177808 |               | success_rate |
| SGD           | ADAMW         |      0   |      0   | ADAMW    | nan        |               | success_rate |
| SGD           | ADAGRAD       |      0   |      0.2 | ADAGRAD  |   0.373901 |               | success_rate |
| SGD           | NOVOGRAD      |      0   |      0   | NOVOGRAD | nan        |               | success_rate |
| ADAMW         | ADAGRAD       |      0   |      0.2 | ADAGRAD  |   0.373901 |               | success_rate |
| ADAMW         | NOVOGRAD      |      0   |      0   | NOVOGRAD | nan        |               | success_rate |
| ADAGRAD       | NOVOGRAD      |      0.2 |      0   | ADAGRAD  |   0.373901 |               | success_rate |

