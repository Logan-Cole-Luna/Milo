# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Success Rate Statistics

Number of runs: 5

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better    |   p-value | Significant   | Metric       |
|:--------------|:--------------|---------:|---------:|:----------|----------:|:--------------|:-------------|
| MILO          | MILO_LW       |      0   |      0.4 | MILO_LW   | 0.177808  |               | success_rate |
| MILO          | SGD           |      0   |      0.2 | SGD       | 0.373901  |               | success_rate |
| MILO          | ADAMW         |      0   |      0.2 | ADAMW     | 0.373901  |               | success_rate |
| MILO          | ADAM_MINI     |      0   |      0.2 | ADAM_MINI | 0.373901  |               | success_rate |
| MILO          | NOVOGRAD      |      0   |      0.6 | NOVOGRAD  | 0.0341094 | *             | success_rate |
| MILO          | ADAGRAD       |      0   |      0.2 | ADAGRAD   | 0.373901  |               | success_rate |
| MILO          | ADEMAMIX      |      0   |      0.4 | ADEMAMIX  | 0.177808  |               | success_rate |
| MILO_LW       | SGD           |      0.4 |      0.2 | MILO_LW   | 0.545424  |               | success_rate |
| MILO_LW       | ADAMW         |      0.4 |      0.2 | MILO_LW   | 0.545424  |               | success_rate |
| MILO_LW       | ADAM_MINI     |      0.4 |      0.2 | MILO_LW   | 0.545424  |               | success_rate |
| MILO_LW       | NOVOGRAD      |      0.4 |      0.6 | NOVOGRAD  | 0.537773  |               | success_rate |
| MILO_LW       | ADAGRAD       |      0.4 |      0.2 | MILO_LW   | 0.545424  |               | success_rate |
| MILO_LW       | ADEMAMIX      |      0.4 |      0.4 | ADEMAMIX  | 1         |               | success_rate |
| SGD           | ADAMW         |      0.2 |      0.2 | ADAMW     | 1         |               | success_rate |
| SGD           | ADAM_MINI     |      0.2 |      0.2 | ADAM_MINI | 1         |               | success_rate |
| SGD           | NOVOGRAD      |      0.2 |      0.6 | NOVOGRAD  | 0.184949  |               | success_rate |
| SGD           | ADAGRAD       |      0.2 |      0.2 | ADAGRAD   | 1         |               | success_rate |
| SGD           | ADEMAMIX      |      0.2 |      0.4 | ADEMAMIX  | 0.545424  |               | success_rate |
| ADAMW         | ADAM_MINI     |      0.2 |      0.2 | ADAM_MINI | 1         |               | success_rate |
| ADAMW         | NOVOGRAD      |      0.2 |      0.6 | NOVOGRAD  | 0.184949  |               | success_rate |
| ADAMW         | ADAGRAD       |      0.2 |      0.2 | ADAGRAD   | 1         |               | success_rate |
| ADAMW         | ADEMAMIX      |      0.2 |      0.4 | ADEMAMIX  | 0.545424  |               | success_rate |
| ADAM_MINI     | NOVOGRAD      |      0.2 |      0.6 | NOVOGRAD  | 0.184949  |               | success_rate |
| ADAM_MINI     | ADAGRAD       |      0.2 |      0.2 | ADAGRAD   | 1         |               | success_rate |
| ADAM_MINI     | ADEMAMIX      |      0.2 |      0.4 | ADEMAMIX  | 0.545424  |               | success_rate |
| NOVOGRAD      | ADAGRAD       |      0.6 |      0.2 | NOVOGRAD  | 0.184949  |               | success_rate |
| NOVOGRAD      | ADEMAMIX      |      0.6 |      0.4 | NOVOGRAD  | 0.537773  |               | success_rate |
| ADAGRAD       | ADEMAMIX      |      0.2 |      0.4 | ADEMAMIX  | 0.545424  |               | success_rate |

