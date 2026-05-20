# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Success Rate Statistics

Number of runs: 5

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better    |     p-value | Significant   | Metric       |
|:--------------|:--------------|---------:|---------:|:----------|------------:|:--------------|:-------------|
| MILO          | MILO_LW       |      1   |      0.6 | MILO      |   0.177808  |               | success_rate |
| MILO          | SGD           |      1   |      0.2 | MILO      |   0.0161301 | *             | success_rate |
| MILO          | ADAMW         |      1   |      1   | ADAMW     | nan         |               | success_rate |
| MILO          | ADAM_MINI     |      1   |      0.8 | MILO      |   0.373901  |               | success_rate |
| MILO          | NOVOGRAD      |      1   |      1   | NOVOGRAD  | nan         |               | success_rate |
| MILO          | ADAGRAD       |      1   |      0.6 | MILO      |   0.177808  |               | success_rate |
| MILO          | ADEMAMIX      |      1   |      0.8 | MILO      |   0.373901  |               | success_rate |
| MILO          | MUON          |      1   |      1   | MUON      | nan         |               | success_rate |
| MILO_LW       | SGD           |      0.6 |      0.2 | MILO_LW   |   0.242876  |               | success_rate |
| MILO_LW       | ADAMW         |      0.6 |      1   | ADAMW     |   0.177808  |               | success_rate |
| MILO_LW       | ADAM_MINI     |      0.6 |      0.8 | ADAM_MINI |   0.545424  |               | success_rate |
| MILO_LW       | NOVOGRAD      |      0.6 |      1   | NOVOGRAD  |   0.177808  |               | success_rate |
| MILO_LW       | ADAGRAD       |      0.6 |      0.6 | ADAGRAD   |   1         |               | success_rate |
| MILO_LW       | ADEMAMIX      |      0.6 |      0.8 | ADEMAMIX  |   0.545424  |               | success_rate |
| MILO_LW       | MUON          |      0.6 |      1   | MUON      |   0.177808  |               | success_rate |
| SGD           | ADAMW         |      0.2 |      1   | ADAMW     |   0.0161301 | *             | success_rate |
| SGD           | ADAM_MINI     |      0.2 |      0.8 | ADAM_MINI |   0.066688  |               | success_rate |
| SGD           | NOVOGRAD      |      0.2 |      1   | NOVOGRAD  |   0.0161301 | *             | success_rate |
| SGD           | ADAGRAD       |      0.2 |      0.6 | ADAGRAD   |   0.242876  |               | success_rate |
| SGD           | ADEMAMIX      |      0.2 |      0.8 | ADEMAMIX  |   0.066688  |               | success_rate |
| SGD           | MUON          |      0.2 |      1   | MUON      |   0.0161301 | *             | success_rate |
| ADAMW         | ADAM_MINI     |      1   |      0.8 | ADAMW     |   0.373901  |               | success_rate |
| ADAMW         | NOVOGRAD      |      1   |      1   | NOVOGRAD  | nan         |               | success_rate |
| ADAMW         | ADAGRAD       |      1   |      0.6 | ADAMW     |   0.177808  |               | success_rate |
| ADAMW         | ADEMAMIX      |      1   |      0.8 | ADAMW     |   0.373901  |               | success_rate |
| ADAMW         | MUON          |      1   |      1   | MUON      | nan         |               | success_rate |
| ADAM_MINI     | NOVOGRAD      |      0.8 |      1   | NOVOGRAD  |   0.373901  |               | success_rate |
| ADAM_MINI     | ADAGRAD       |      0.8 |      0.6 | ADAM_MINI |   0.545424  |               | success_rate |
| ADAM_MINI     | ADEMAMIX      |      0.8 |      0.8 | ADEMAMIX  |   1         |               | success_rate |
| ADAM_MINI     | MUON          |      0.8 |      1   | MUON      |   0.373901  |               | success_rate |
| NOVOGRAD      | ADAGRAD       |      1   |      0.6 | NOVOGRAD  |   0.177808  |               | success_rate |
| NOVOGRAD      | ADEMAMIX      |      1   |      0.8 | NOVOGRAD  |   0.373901  |               | success_rate |
| NOVOGRAD      | MUON          |      1   |      1   | MUON      | nan         |               | success_rate |
| ADAGRAD       | ADEMAMIX      |      0.6 |      0.8 | ADEMAMIX  |   0.545424  |               | success_rate |
| ADAGRAD       | MUON          |      0.6 |      1   | MUON      |   0.177808  |               | success_rate |
| ADEMAMIX      | MUON          |      0.8 |      1   | MUON      |   0.373901  |               | success_rate |

