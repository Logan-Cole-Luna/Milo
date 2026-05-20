# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.7207 | 0.0296 | 0.0133 | 0.6839 | 0.7575 |
| MILO_LW | 0.6879 | 0.0335 | 0.0150 | 0.6463 | 0.7295 |
| SGD | 1.1963 | 0.0566 | 0.0253 | 1.1260 | 1.2666 |
| ADAMW | 0.6226 | 0.0370 | 0.0166 | 0.5766 | 0.6686 |
| ADAM_MINI | 1.5347 | 0.0100 | 0.0045 | 1.5222 | 1.5471 |
| NOVOGRAD | 1.5215 | 0.0465 | 0.0208 | 1.4639 | 1.5792 |
| ADAGRAD | 1.1922 | 0.0192 | 0.0086 | 1.1683 | 1.2161 |
| ADEMAMIX | 0.7462 | 0.0355 | 0.0159 | 0.7021 | 0.7903 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.720657 | 0.687874 | MILO_LW  | 0.140429    |               | final_validation_loss |
| MILO          | SGD           | 0.720657 | 1.19633  | MILO     | 2.83256e-06 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.720657 | 0.62256  | ADAMW    | 0.00192647  | **            | final_validation_loss |
| MILO          | ADAM_MINI     | 0.720657 | 1.53467  | MILO     | 3.7718e-08  | ***           | final_validation_loss |
| MILO          | NOVOGRAD      | 0.720657 | 1.52153  | MILO     | 1.03941e-08 | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.720657 | 1.19219  | MILO     | 1.61458e-08 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      | 0.720657 | 0.746198 | MILO     | 0.25299     |               | final_validation_loss |
| MILO_LW       | SGD           | 0.687874 | 1.19633  | MILO_LW  | 1.13033e-06 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         | 0.687874 | 0.62256  | ADAMW    | 0.0193829   | *             | final_validation_loss |
| MILO_LW       | ADAM_MINI     | 0.687874 | 1.53467  | MILO_LW  | 9.18561e-08 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.687874 | 1.52153  | MILO_LW  | 3.78101e-09 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.687874 | 1.19219  | MILO_LW  | 4.94768e-08 | ***           | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.687874 | 0.746198 | MILO_LW  | 0.0283742   | *             | final_validation_loss |
| SGD           | ADAMW         | 1.19633  | 0.62256  | ADAMW    | 3.32898e-07 | ***           | final_validation_loss |
| SGD           | ADAM_MINI     | 1.19633  | 1.53467  | SGD      | 0.000132661 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 1.19633  | 1.52153  | SGD      | 1.16539e-05 | ***           | final_validation_loss |
| SGD           | ADAGRAD       | 1.19633  | 1.19219  | ADAGRAD  | 0.883168    |               | final_validation_loss |
| SGD           | ADEMAMIX      | 1.19633  | 0.746198 | ADEMAMIX | 1.98079e-06 | ***           | final_validation_loss |
| ADAMW         | ADAM_MINI     | 0.62256  | 1.53467  | ADAMW    | 1.43423e-07 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.62256  | 1.52153  | ADAMW    | 1.38387e-09 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.62256  | 1.19219  | ADAMW    | 8.05489e-08 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.62256  | 0.746198 | ADAMW    | 0.000659498 | ***           | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      | 1.53467  | 1.52153  | NOVOGRAD | 0.56705     |               | final_validation_loss |
| ADAM_MINI     | ADAGRAD       | 1.53467  | 1.19219  | ADAGRAD  | 3.29239e-08 | ***           | final_validation_loss |
| ADAM_MINI     | ADEMAMIX      | 1.53467  | 0.746198 | ADEMAMIX | 2.03153e-07 | ***           | final_validation_loss |
| NOVOGRAD      | ADAGRAD       | 1.52153  | 1.19219  | ADAGRAD  | 1.64319e-05 | ***           | final_validation_loss |
| NOVOGRAD      | ADEMAMIX      | 1.52153  | 0.746198 | ADEMAMIX | 4.91685e-09 | ***           | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 1.19219  | 0.746198 | ADEMAMIX | 2.13943e-07 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 80.5867 | 0.5427 | 0.2427 | 79.9128 | 81.2605 |
| MILO_LW | 80.7852 | 0.5419 | 0.2423 | 80.1123 | 81.4581 |
| SGD | 72.0711 | 0.7644 | 0.3418 | 71.1220 | 73.0202 |
| ADAMW | 82.1689 | 1.1617 | 0.5195 | 80.7264 | 83.6114 |
| ADAM_MINI | 43.3896 | 0.4161 | 0.1861 | 42.8730 | 43.9062 |
| NOVOGRAD | 62.1600 | 0.4674 | 0.2090 | 61.5797 | 62.7403 |
| ADAGRAD | 75.8993 | 0.3483 | 0.1558 | 75.4668 | 76.3317 |
| ADEMAMIX | 82.3556 | 0.4056 | 0.1814 | 81.8520 | 82.8592 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  80.5867 |  80.7852 | MILO_LW  | 0.578643    |               | final_validation_accuracy |
| MILO          | SGD           |  80.5867 |  72.0711 | MILO     | 1.24166e-07 | ***           | final_validation_accuracy |
| MILO          | ADAMW         |  80.5867 |  82.1689 | ADAMW    | 0.0349095   | *             | final_validation_accuracy |
| MILO          | ADAM_MINI     |  80.5867 |  43.3896 | MILO     | 1.25974e-13 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  80.5867 |  62.16   | MILO     | 1.44238e-11 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  80.5867 |  75.8993 | MILO     | 1.05137e-06 | ***           | final_validation_accuracy |
| MILO          | ADEMAMIX      |  80.5867 |  82.3556 | ADEMAMIX | 0.00051811  | ***           | final_validation_accuracy |
| MILO_LW       | SGD           |  80.7852 |  72.0711 | MILO_LW  | 1.05984e-07 | ***           | final_validation_accuracy |
| MILO_LW       | ADAMW         |  80.7852 |  82.1689 | ADAMW    | 0.0547671   |               | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  80.7852 |  43.3896 | MILO_LW  | 1.18312e-13 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  80.7852 |  62.16   | MILO_LW  | 1.30746e-11 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  80.7852 |  75.8993 | MILO_LW  | 7.85365e-07 | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  80.7852 |  82.3556 | ADEMAMIX | 0.00106259  | **            | final_validation_accuracy |
| SGD           | ADAMW         |  72.0711 |  82.1689 | ADAMW    | 9.2012e-07  | ***           | final_validation_accuracy |
| SGD           | ADAM_MINI     |  72.0711 |  43.3896 | SGD      | 2.46936e-10 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  72.0711 |  62.16   | SGD      | 8.94502e-08 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  72.0711 |  75.8993 | ADAGRAD  | 8.04932e-05 | ***           | final_validation_accuracy |
| SGD           | ADEMAMIX      |  72.0711 |  82.3556 | ADEMAMIX | 1.57997e-07 | ***           | final_validation_accuracy |
| ADAMW         | ADAM_MINI     |  82.1689 |  43.3896 | ADAMW    | 1.07404e-08 | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  82.1689 |  62.16   | ADAMW    | 1.74766e-07 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  82.1689 |  75.8993 | ADAMW    | 0.000122771 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  82.1689 |  82.3556 | ADEMAMIX | 0.748339    |               | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  43.3896 |  62.16   | NOVOGRAD | 3.62452e-12 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  43.3896 |  75.8993 | ADAGRAD  | 2.45281e-14 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADEMAMIX      |  43.3896 |  82.3556 | ADEMAMIX | 4.45536e-15 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADAGRAD       |  62.16   |  75.8993 | ADAGRAD  | 8.47103e-11 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADEMAMIX      |  62.16   |  82.3556 | ADEMAMIX | 2.14609e-12 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  75.8993 |  82.3556 | ADEMAMIX | 5.28158e-09 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.8056 | 0.0051 | 0.0023 | 0.7992 | 0.8120 |
| MILO_LW | 0.8078 | 0.0055 | 0.0024 | 0.8010 | 0.8146 |
| SGD | 0.7182 | 0.0068 | 0.0030 | 0.7097 | 0.7266 |
| ADAMW | 0.8204 | 0.0120 | 0.0054 | 0.8055 | 0.8353 |
| ADAM_MINI | 0.4250 | 0.0050 | 0.0022 | 0.4188 | 0.4312 |
| NOVOGRAD | 0.6173 | 0.0066 | 0.0030 | 0.6091 | 0.6255 |
| ADAGRAD | 0.7582 | 0.0031 | 0.0014 | 0.7544 | 0.7620 |
| ADEMAMIX | 0.8231 | 0.0045 | 0.0020 | 0.8174 | 0.8287 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.805605 | 0.807797 | MILO_LW  | 0.531016    |               | final_validation_f1_score |
| MILO          | SGD           | 0.805605 | 0.718173 | MILO     | 3.67291e-08 | ***           | final_validation_f1_score |
| MILO          | ADAMW         | 0.805605 | 0.820402 | ADAMW    | 0.0488744   | *             | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.805605 | 0.424986 | MILO     | 2.87093e-14 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.805605 | 0.617296 | MILO     | 8.49579e-11 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.805605 | 0.758229 | MILO     | 8.99985e-07 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.805605 | 0.823051 | ADEMAMIX | 0.000478284 | ***           | final_validation_f1_score |
| MILO_LW       | SGD           | 0.807797 | 0.718173 | MILO_LW  | 2.5527e-08  | ***           | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.807797 | 0.820402 | ADAMW    | 0.0801816   |               | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.807797 | 0.424986 | MILO_LW  | 4.25108e-14 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.807797 | 0.617296 | MILO_LW  | 5.8405e-11  | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.807797 | 0.758229 | MILO_LW  | 1.33159e-06 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.807797 | 0.823051 | ADEMAMIX | 0.00147611  | **            | final_validation_f1_score |
| SGD           | ADAMW         | 0.718173 | 0.820402 | ADAMW    | 1.90246e-06 | ***           | final_validation_f1_score |
| SGD           | ADAM_MINI     | 0.718173 | 0.424986 | SGD      | 5.63514e-12 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.718173 | 0.617296 | SGD      | 1.06487e-08 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.718173 | 0.758229 | ADAGRAD  | 3.56234e-05 | ***           | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.718173 | 0.823051 | ADEMAMIX | 1.7578e-08  | ***           | final_validation_f1_score |
| ADAMW         | ADAM_MINI     | 0.820402 | 0.424986 | ADAMW    | 4.59038e-09 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.820402 | 0.617296 | ADAMW    | 3.17029e-08 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.820402 | 0.758229 | ADAMW    | 0.000182475 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.820402 | 0.823051 | ADEMAMIX | 0.663913    |               | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.424986 | 0.617296 | NOVOGRAD | 8.13742e-11 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.424986 | 0.758229 | ADAGRAD  | 1.93482e-12 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADEMAMIX      | 0.424986 | 0.823051 | ADEMAMIX | 1.63946e-14 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADAGRAD       | 0.617296 | 0.758229 | ADAGRAD  | 2.50065e-08 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADEMAMIX      | 0.617296 | 0.823051 | ADEMAMIX | 1.02407e-10 | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.758229 | 0.823051 | ADEMAMIX | 2.73528e-08 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9783 | 0.0008 | 0.0004 | 0.9773 | 0.9793 |
| MILO_LW | 0.9794 | 0.0013 | 0.0006 | 0.9778 | 0.9811 |
| SGD | 0.9586 | 0.0027 | 0.0012 | 0.9552 | 0.9619 |
| ADAMW | 0.9832 | 0.0015 | 0.0007 | 0.9813 | 0.9851 |
| ADAM_MINI | 0.8608 | 0.0024 | 0.0011 | 0.8579 | 0.8637 |
| NOVOGRAD | 0.9295 | 0.0016 | 0.0007 | 0.9275 | 0.9315 |
| ADAGRAD | 0.9637 | 0.0010 | 0.0005 | 0.9625 | 0.9650 |
| ADEMAMIX | 0.9831 | 0.0006 | 0.0003 | 0.9824 | 0.9839 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.978291 | 0.979444 | MILO_LW  | 0.136412    |               | final_validation_auc |
| MILO          | SGD           | 0.978291 | 0.958558 | MILO     | 3.00236e-05 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.978291 | 0.983191 | ADAMW    | 0.00076656  | ***           | final_validation_auc |
| MILO          | ADAM_MINI     | 0.978291 | 0.860801 | MILO     | 1.97957e-09 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.978291 | 0.929522 | MILO     | 1.80161e-09 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.978291 | 0.963738 | MILO     | 1.69182e-08 | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.978291 | 0.983133 | ADEMAMIX | 7.71963e-06 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.979444 | 0.958558 | MILO_LW  | 5.96428e-06 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.979444 | 0.983191 | ADAMW    | 0.00344778  | **            | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.979444 | 0.860801 | MILO_LW  | 3.4339e-11  | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.979444 | 0.929522 | MILO_LW  | 3.26479e-11 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.979444 | 0.963738 | MILO_LW  | 4.76166e-08 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.979444 | 0.983133 | ADEMAMIX | 0.00141308  | **            | final_validation_auc |
| SGD           | ADAMW         | 0.958558 | 0.983191 | ADAMW    | 1.08678e-06 | ***           | final_validation_auc |
| SGD           | ADAM_MINI     | 0.958558 | 0.860801 | SGD      | 7.99071e-12 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.958558 | 0.929522 | SGD      | 3.35408e-07 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.958558 | 0.963738 | ADAGRAD  | 0.00940093  | **            | final_validation_auc |
| SGD           | ADEMAMIX      | 0.958558 | 0.983133 | ADEMAMIX | 1.65483e-05 | ***           | final_validation_auc |
| ADAMW         | ADAM_MINI     | 0.983191 | 0.860801 | ADAMW    | 4.07463e-12 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.983191 | 0.929522 | ADAMW    | 1.56224e-11 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.983191 | 0.963738 | ADAMW    | 7.06777e-08 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.983191 | 0.983133 | ADAMW    | 0.941002    |               | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.860801 | 0.929522 | NOVOGRAD | 1.75613e-10 | ***           | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.860801 | 0.963738 | ADAGRAD  | 6.63078e-10 | ***           | final_validation_auc |
| ADAM_MINI     | ADEMAMIX      | 0.860801 | 0.983133 | ADEMAMIX | 5.02308e-09 | ***           | final_validation_auc |
| NOVOGRAD      | ADAGRAD       | 0.929522 | 0.963738 | ADAGRAD  | 2.10052e-09 | ***           | final_validation_auc |
| NOVOGRAD      | ADEMAMIX      | 0.929522 | 0.983133 | ADEMAMIX | 6.28612e-09 | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.963738 | 0.983133 | ADEMAMIX | 8.73385e-09 | ***           | final_validation_auc |

