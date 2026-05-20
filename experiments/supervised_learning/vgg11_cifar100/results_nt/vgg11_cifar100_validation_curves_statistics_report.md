# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 2.5159 | 0.0282 | 0.0126 | 2.4808 | 2.5509 |
| MILO_LW | 2.5363 | 0.0443 | 0.0198 | 2.4813 | 2.5912 |
| SGD | 2.9854 | 0.1016 | 0.0454 | 2.8593 | 3.1115 |
| ADAMW | 2.4211 | 0.0553 | 0.0247 | 2.3524 | 2.4898 |
| ADAM_MINI | 4.1739 | 0.1073 | 0.0480 | 4.0407 | 4.3071 |
| NOVOGRAD | 2.8526 | 0.0224 | 0.0100 | 2.8248 | 2.8804 |
| ADAGRAD | 3.7819 | 0.0866 | 0.0387 | 3.6744 | 3.8894 |
| ADEMAMIX | 2.7651 | 0.0728 | 0.0326 | 2.6747 | 2.8555 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  2.51585 |  2.53627 | MILO     | 0.41413     |               | final_validation_loss |
| MILO          | SGD           |  2.51585 |  2.98536 | MILO     | 0.000271485 | ***           | final_validation_loss |
| MILO          | ADAMW         |  2.51585 |  2.4211  | ADAMW    | 0.0144774   | *             | final_validation_loss |
| MILO          | ADAM_MINI     |  2.51585 |  4.17387 | MILO     | 1.28055e-06 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      |  2.51585 |  2.85261 | MILO     | 5.38377e-08 | ***           | final_validation_loss |
| MILO          | ADAGRAD       |  2.51585 |  3.78191 | MILO     | 9.23409e-07 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      |  2.51585 |  2.76507 | MILO     | 0.00072552  | ***           | final_validation_loss |
| MILO_LW       | SGD           |  2.53627 |  2.98536 | MILO_LW  | 0.000169717 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         |  2.53627 |  2.4211  | ADAMW    | 0.00719173  | **            | final_validation_loss |
| MILO_LW       | ADAM_MINI     |  2.53627 |  4.17387 | MILO_LW  | 2.92093e-07 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      |  2.53627 |  2.85261 | MILO_LW  | 8.3156e-06  | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       |  2.53627 |  3.78191 | MILO_LW  | 1.3068e-07  | ***           | final_validation_loss |
| MILO_LW       | ADEMAMIX      |  2.53627 |  2.76507 | MILO_LW  | 0.000673939 | ***           | final_validation_loss |
| SGD           | ADAMW         |  2.98536 |  2.4211  | ADAMW    | 2.87506e-05 | ***           | final_validation_loss |
| SGD           | ADAM_MINI     |  2.98536 |  4.17387 | SGD      | 9.67001e-08 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      |  2.98536 |  2.85261 | NOVOGRAD | 0.0414422   | *             | final_validation_loss |
| SGD           | ADAGRAD       |  2.98536 |  3.78191 | SGD      | 1.19013e-06 | ***           | final_validation_loss |
| SGD           | ADEMAMIX      |  2.98536 |  2.76507 | ADEMAMIX | 0.00520615  | **            | final_validation_loss |
| ADAMW         | ADAM_MINI     |  2.4211  |  4.17387 | ADAMW    | 5.83385e-08 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      |  2.4211  |  2.85261 | ADAMW    | 1.07175e-05 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       |  2.4211  |  3.78191 | ADAMW    | 1.92317e-08 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      |  2.4211  |  2.76507 | ADAMW    | 4.57143e-05 | ***           | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      |  4.17387 |  2.85261 | NOVOGRAD | 5.25683e-06 | ***           | final_validation_loss |
| ADAM_MINI     | ADAGRAD       |  4.17387 |  3.78191 | ADAGRAD  | 0.000263029 | ***           | final_validation_loss |
| ADAM_MINI     | ADEMAMIX      |  4.17387 |  2.76507 | ADEMAMIX | 4.73851e-08 | ***           | final_validation_loss |
| NOVOGRAD      | ADAGRAD       |  2.85261 |  3.78191 | NOVOGRAD | 6.89742e-06 | ***           | final_validation_loss |
| NOVOGRAD      | ADEMAMIX      |  2.85261 |  2.76507 | ADEMAMIX | 0.052537    |               | final_validation_loss |
| ADAGRAD       | ADEMAMIX      |  3.78191 |  2.76507 | ADEMAMIX | 5.57715e-08 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 34.9363 | 0.8153 | 0.3646 | 33.9240 | 35.9486 |
| MILO_LW | 34.8030 | 1.0698 | 0.4784 | 33.4746 | 36.1313 |
| SGD | 34.6874 | 0.7443 | 0.3329 | 33.7632 | 35.6116 |
| ADAMW | 35.7422 | 0.5749 | 0.2571 | 35.0284 | 36.4560 |
| ADAM_MINI | 5.2830 | 1.2382 | 0.5537 | 3.7455 | 6.8204 |
| NOVOGRAD | 27.9230 | 0.5977 | 0.2673 | 27.1808 | 28.6651 |
| ADAGRAD | 35.7244 | 0.7671 | 0.3430 | 34.7720 | 36.6769 |
| ADEMAMIX | 26.6548 | 1.9708 | 0.8814 | 24.2077 | 29.1019 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 34.9363  | 34.803   | MILO     | 0.830519    |               | final_validation_accuracy |
| MILO          | SGD           | 34.9363  | 34.6874  | MILO     | 0.627877    |               | final_validation_accuracy |
| MILO          | ADAMW         | 34.9363  | 35.7422  | ADAMW    | 0.11269     |               | final_validation_accuracy |
| MILO          | ADAM_MINI     | 34.9363  |  5.28296 | MILO     | 8.84699e-10 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      | 34.9363  | 27.923   | MILO     | 7.10234e-07 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       | 34.9363  | 35.7244  | ADAGRAD  | 0.154197    |               | final_validation_accuracy |
| MILO          | ADEMAMIX      | 34.9363  | 26.6548  | MILO     | 0.000241924 | ***           | final_validation_accuracy |
| MILO_LW       | SGD           | 34.803   | 34.6874  | MILO_LW  | 0.84837     |               | final_validation_accuracy |
| MILO_LW       | ADAMW         | 34.803   | 35.7422  | ADAMW    | 0.133414    |               | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     | 34.803   |  5.28296 | MILO_LW  | 2.26635e-10 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      | 34.803   | 27.923   | MILO_LW  | 1.11328e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       | 34.803   | 35.7244  | ADAGRAD  | 0.160015    |               | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      | 34.803   | 26.6548  | MILO_LW  | 0.000161783 | ***           | final_validation_accuracy |
| SGD           | ADAMW         | 34.6874  | 35.7422  | ADAMW    | 0.0382814   | *             | final_validation_accuracy |
| SGD           | ADAM_MINI     | 34.6874  |  5.28296 | SGD      | 1.89813e-09 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      | 34.6874  | 27.923   | SGD      | 4.03468e-07 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       | 34.6874  | 35.7244  | ADAGRAD  | 0.0618927   |               | final_validation_accuracy |
| SGD           | ADEMAMIX      | 34.6874  | 26.6548  | SGD      | 0.000325211 | ***           | final_validation_accuracy |
| ADAMW         | ADAM_MINI     | 35.7422  |  5.28296 | ADAMW    | 1.08993e-08 | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      | 35.7422  | 27.923   | ADAMW    | 2.74238e-08 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       | 35.7422  | 35.7244  | ADAMW    | 0.968016    |               | final_validation_accuracy |
| ADAMW         | ADEMAMIX      | 35.7422  | 26.6548  | ADAMW    | 0.000259878 | ***           | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  5.28296 | 27.923   | NOVOGRAD | 4.56588e-08 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  5.28296 | 35.7244  | ADAGRAD  | 1.18926e-09 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADEMAMIX      |  5.28296 | 26.6548  | ADEMAMIX | 2.532e-07   | ***           | final_validation_accuracy |
| NOVOGRAD      | ADAGRAD       | 27.923   | 35.7244  | ADAGRAD  | 1.83339e-07 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADEMAMIX      | 27.923   | 26.6548  | NOVOGRAD | 0.230142    |               | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      | 35.7244  | 26.6548  | ADAGRAD  | 0.000171027 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.3386 | 0.0103 | 0.0046 | 0.3258 | 0.3514 |
| MILO_LW | 0.3376 | 0.0114 | 0.0051 | 0.3234 | 0.3518 |
| SGD | 0.3396 | 0.0105 | 0.0047 | 0.3266 | 0.3527 |
| ADAMW | 0.3406 | 0.0061 | 0.0027 | 0.3330 | 0.3483 |
| ADAM_MINI | 0.0259 | 0.0109 | 0.0049 | 0.0123 | 0.0394 |
| NOVOGRAD | 0.2509 | 0.0082 | 0.0037 | 0.2407 | 0.2611 |
| ADAGRAD | 0.3597 | 0.0090 | 0.0040 | 0.3485 | 0.3708 |
| ADEMAMIX | 0.2420 | 0.0234 | 0.0104 | 0.2130 | 0.2710 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |    Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|----------:|----------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.338611  | 0.337642  | MILO     | 0.891612    |               | final_validation_f1_score |
| MILO          | SGD           | 0.338611  | 0.339646  | SGD      | 0.879       |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.338611  | 0.340638  | ADAMW    | 0.717968    |               | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.338611  | 0.0258557 | MILO     | 5.25991e-11 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.338611  | 0.250858  | MILO     | 6.69641e-07 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.338611  | 0.359665  | ADAGRAD  | 0.00908734  | **            | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.338611  | 0.242032  | MILO     | 0.000234381 | ***           | final_validation_f1_score |
| MILO_LW       | SGD           | 0.337642  | 0.339646  | SGD      | 0.780168    |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.337642  | 0.340638  | ADAMW    | 0.623834    |               | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.337642  | 0.0258557 | MILO_LW  | 8.0086e-11  | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.337642  | 0.250858  | MILO_LW  | 1.81042e-06 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.337642  | 0.359665  | ADAGRAD  | 0.0103631   | *             | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.337642  | 0.242032  | MILO_LW  | 0.000205964 | ***           | final_validation_f1_score |
| SGD           | ADAMW         | 0.339646  | 0.340638  | ADAMW    | 0.86091     |               | final_validation_f1_score |
| SGD           | ADAM_MINI     | 0.339646  | 0.0258557 | SGD      | 5.30114e-11 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.339646  | 0.250858  | SGD      | 7.06769e-07 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.339646  | 0.359665  | ADAGRAD  | 0.0122602   | *             | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.339646  | 0.242032  | SGD      | 0.00021543  | ***           | final_validation_f1_score |
| ADAMW         | ADAM_MINI     | 0.340638  | 0.0258557 | ADAMW    | 9.30986e-10 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.340638  | 0.250858  | ADAMW    | 1.2104e-07  | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.340638  | 0.359665  | ADAGRAD  | 0.00571145  | **            | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.340638  | 0.242032  | ADAMW    | 0.000428906 | ***           | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.0258557 | 0.250858  | NOVOGRAD | 1.07551e-09 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.0258557 | 0.359665  | ADAGRAD  | 3.66858e-11 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADEMAMIX      | 0.0258557 | 0.242032  | ADEMAMIX | 2.58866e-06 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADAGRAD       | 0.250858  | 0.359665  | ADAGRAD  | 4.5178e-08  | ***           | final_validation_f1_score |
| NOVOGRAD      | ADEMAMIX      | 0.250858  | 0.242032  | NOVOGRAD | 0.461804    |               | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.359665  | 0.242032  | ADAGRAD  | 0.000111965 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9404 | 0.0020 | 0.0009 | 0.9378 | 0.9429 |
| MILO_LW | 0.9383 | 0.0033 | 0.0015 | 0.9342 | 0.9424 |
| SGD | 0.9368 | 0.0036 | 0.0016 | 0.9324 | 0.9412 |
| ADAMW | 0.9465 | 0.0031 | 0.0014 | 0.9426 | 0.9504 |
| ADAM_MINI | 0.7460 | 0.0245 | 0.0110 | 0.7155 | 0.7764 |
| NOVOGRAD | 0.9185 | 0.0016 | 0.0007 | 0.9165 | 0.9205 |
| ADAGRAD | 0.9292 | 0.0021 | 0.0009 | 0.9266 | 0.9319 |
| ADEMAMIX | 0.9273 | 0.0042 | 0.0019 | 0.9220 | 0.9325 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.940352 | 0.938296 | MILO     | 0.278611    |               | final_validation_auc |
| MILO          | SGD           | 0.940352 | 0.936799 | MILO     | 0.0981211   |               | final_validation_auc |
| MILO          | ADAMW         | 0.940352 | 0.946469 | ADAMW    | 0.0084567   | **            | final_validation_auc |
| MILO          | ADAM_MINI     | 0.940352 | 0.74596  | MILO     | 5.44396e-05 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.940352 | 0.918505 | MILO     | 1.27829e-07 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.940352 | 0.929236 | MILO     | 2.95426e-05 | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.940352 | 0.927276 | MILO     | 0.000898919 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.938296 | 0.936799 | MILO_LW  | 0.511114    |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.938296 | 0.946469 | ADAMW    | 0.003979    | **            | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.938296 | 0.74596  | MILO_LW  | 4.94666e-05 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.938296 | 0.918505 | MILO_LW  | 2.73204e-05 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.938296 | 0.929236 | MILO_LW  | 0.00145776  | **            | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.938296 | 0.927276 | MILO_LW  | 0.00203207  | **            | final_validation_auc |
| SGD           | ADAMW         | 0.936799 | 0.946469 | ADAMW    | 0.00193582  | **            | final_validation_auc |
| SGD           | ADAM_MINI     | 0.936799 | 0.74596  | SGD      | 4.9423e-05  | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.936799 | 0.918505 | SGD      | 7.30629e-05 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.936799 | 0.929236 | SGD      | 0.00542746  | **            | final_validation_auc |
| SGD           | ADEMAMIX      | 0.936799 | 0.927276 | SGD      | 0.00505987  | **            | final_validation_auc |
| ADAMW         | ADAM_MINI     | 0.946469 | 0.74596  | ADAMW    | 4.27023e-05 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.946469 | 0.918505 | ADAMW    | 2.30693e-06 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.946469 | 0.929236 | ADAMW    | 1.89572e-05 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.946469 | 0.927276 | ADAMW    | 5.8831e-05  | ***           | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.74596  | 0.918505 | NOVOGRAD | 9.05773e-05 | ***           | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.74596  | 0.929236 | ADAGRAD  | 6.85485e-05 | ***           | final_validation_auc |
| ADAM_MINI     | ADEMAMIX      | 0.74596  | 0.927276 | ADEMAMIX | 5.53466e-05 | ***           | final_validation_auc |
| NOVOGRAD      | ADAGRAD       | 0.918505 | 0.929236 | ADAGRAD  | 2.80448e-05 | ***           | final_validation_auc |
| NOVOGRAD      | ADEMAMIX      | 0.918505 | 0.927276 | ADEMAMIX | 0.00694695  | **            | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.929236 | 0.927276 | ADAGRAD  | 0.389035    |               | final_validation_auc |

