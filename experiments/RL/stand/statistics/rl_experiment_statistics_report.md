# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Success Rate Statistics

Number of runs: 5

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |    p-value | Significant   | Metric       |
|:--------------|:--------------|---------:|---------:|:---------|-----------:|:--------------|:-------------|
| MILO          | MILO_LW       |      1   |      1   | MILO_LW  | nan        |               | success_rate |
| MILO          | SGD           |      1   |      0.6 | MILO     |   0.177808 |               | success_rate |
| MILO          | ADAMW         |      1   |      1   | ADAMW    | nan        |               | success_rate |
| MILO          | ADAGRAD       |      1   |      1   | ADAGRAD  | nan        |               | success_rate |
| MILO          | NOVOGRAD      |      1   |      0.8 | MILO     |   0.373901 |               | success_rate |
| MILO_LW       | SGD           |      1   |      0.6 | MILO_LW  |   0.177808 |               | success_rate |
| MILO_LW       | ADAMW         |      1   |      1   | ADAMW    | nan        |               | success_rate |
| MILO_LW       | ADAGRAD       |      1   |      1   | ADAGRAD  | nan        |               | success_rate |
| MILO_LW       | NOVOGRAD      |      1   |      0.8 | MILO_LW  |   0.373901 |               | success_rate |
| SGD           | ADAMW         |      0.6 |      1   | ADAMW    |   0.177808 |               | success_rate |
| SGD           | ADAGRAD       |      0.6 |      1   | ADAGRAD  |   0.177808 |               | success_rate |
| SGD           | NOVOGRAD      |      0.6 |      0.8 | NOVOGRAD |   0.545424 |               | success_rate |
| ADAMW         | ADAGRAD       |      1   |      1   | ADAGRAD  | nan        |               | success_rate |
| ADAMW         | NOVOGRAD      |      1   |      0.8 | ADAMW    |   0.373901 |               | success_rate |
| ADAGRAD       | NOVOGRAD      |      1   |      0.8 | ADAGRAD  |   0.373901 |               | success_rate |

