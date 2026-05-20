# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 2.0325 | 0.0271 | 0.0121 | 1.9988 | 2.0662 |
| MILO_LW | 2.0706 | 0.0204 | 0.0091 | 2.0453 | 2.0960 |
| SGD | 2.7330 | 0.1245 | 0.0557 | 2.5785 | 2.8875 |
| ADAMW | 2.3730 | 0.1377 | 0.0616 | 2.2020 | 2.5441 |
| ADAM_MINI | 3.6329 | 0.0156 | 0.0070 | 3.6136 | 3.6523 |
| NOVOGRAD | 2.7028 | 0.0316 | 0.0141 | 2.6636 | 2.7420 |
| ADAGRAD | 3.4562 | 0.0396 | 0.0177 | 3.4070 | 3.5054 |
| ADEMAMIX | 2.7144 | 0.0826 | 0.0369 | 2.6119 | 2.8170 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  2.03245 |  2.07064 | MILO     | 0.0382984   | *             | final_validation_loss |
| MILO          | SGD           |  2.03245 |  2.73301 | MILO     | 0.000146316 | ***           | final_validation_loss |
| MILO          | ADAMW         |  2.03245 |  2.37305 | MILO     | 0.00450369  | **            | final_validation_loss |
| MILO          | ADAM_MINI     |  2.03245 |  3.63294 | MILO     | 8.08951e-12 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      |  2.03245 |  2.70284 | MILO     | 5.63283e-10 | ***           | final_validation_loss |
| MILO          | ADAGRAD       |  2.03245 |  3.45619 | MILO     | 3.80954e-11 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      |  2.03245 |  2.71444 | MILO     | 1.40595e-05 | ***           | final_validation_loss |
| MILO_LW       | SGD           |  2.07064 |  2.73301 | MILO_LW  | 0.000222932 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         |  2.07064 |  2.37305 | MILO_LW  | 0.00743812  | **            | final_validation_loss |
| MILO_LW       | ADAM_MINI     |  2.07064 |  3.63294 | MILO_LW  | 5.67221e-14 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      |  2.07064 |  2.70284 | MILO_LW  | 3.43201e-09 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       |  2.07064 |  3.45619 | MILO_LW  | 6.27138e-10 | ***           | final_validation_loss |
| MILO_LW       | ADEMAMIX      |  2.07064 |  2.71444 | MILO_LW  | 3.08041e-05 | ***           | final_validation_loss |
| SGD           | ADAMW         |  2.73301 |  2.37305 | ADAMW    | 0.00255113  | **            | final_validation_loss |
| SGD           | ADAM_MINI     |  2.73301 |  3.63294 | SGD      | 7.12823e-05 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      |  2.73301 |  2.70284 | NOVOGRAD | 0.624057    |               | final_validation_loss |
| SGD           | ADAGRAD       |  2.73301 |  3.45619 | SGD      | 7.92912e-05 | ***           | final_validation_loss |
| SGD           | ADEMAMIX      |  2.73301 |  2.71444 | ADEMAMIX | 0.789031    |               | final_validation_loss |
| ADAMW         | ADAM_MINI     |  2.37305 |  3.63294 | ADAMW    | 2.83647e-05 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      |  2.37305 |  2.70284 | ADAMW    | 0.00486898  | **            | final_validation_loss |
| ADAMW         | ADAGRAD       |  2.37305 |  3.45619 | ADAMW    | 2.32964e-05 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      |  2.37305 |  2.71444 | ADAMW    | 0.00248794  | **            | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      |  3.63294 |  2.70284 | NOVOGRAD | 2.42148e-09 | ***           | final_validation_loss |
| ADAM_MINI     | ADAGRAD       |  3.63294 |  3.45619 | ADAGRAD  | 0.000195766 | ***           | final_validation_loss |
| ADAM_MINI     | ADEMAMIX      |  3.63294 |  2.71444 | ADEMAMIX | 9.12157e-06 | ***           | final_validation_loss |
| NOVOGRAD      | ADAGRAD       |  2.70284 |  3.45619 | NOVOGRAD | 1.59648e-09 | ***           | final_validation_loss |
| NOVOGRAD      | ADEMAMIX      |  2.70284 |  2.71444 | NOVOGRAD | 0.780719    |               | final_validation_loss |
| ADAGRAD       | ADEMAMIX      |  3.45619 |  2.71444 | ADEMAMIX | 2.71905e-06 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 50.0444 | 0.4207 | 0.1882 | 49.5220 | 50.5668 |
| MILO_LW | 48.8948 | 0.3535 | 0.1581 | 48.4559 | 49.3337 |
| SGD | 34.0207 | 2.3869 | 1.0675 | 31.0570 | 36.9845 |
| ADAMW | 48.0119 | 1.8122 | 0.8104 | 45.7617 | 50.2620 |
| ADAM_MINI | 14.9630 | 0.2225 | 0.0995 | 14.6867 | 15.2392 |
| NOVOGRAD | 33.9467 | 0.7772 | 0.3476 | 32.9816 | 34.9117 |
| ADAGRAD | 18.0504 | 0.6639 | 0.2969 | 17.2260 | 18.8747 |
| ADEMAMIX | 48.9333 | 1.6508 | 0.7383 | 46.8836 | 50.9831 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  50.0444 |  48.8948 | MILO     | 0.00171537  | **            | final_validation_accuracy |
| MILO          | SGD           |  50.0444 |  34.0207 | MILO     | 8.17478e-05 | ***           | final_validation_accuracy |
| MILO          | ADAMW         |  50.0444 |  48.0119 | MILO     | 0.0648543   |               | final_validation_accuracy |
| MILO          | ADAM_MINI     |  50.0444 |  14.963  | MILO     | 2.53686e-12 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  50.0444 |  33.9467 | MILO     | 1.00309e-08 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  50.0444 |  18.0504 | MILO     | 1.05325e-11 | ***           | final_validation_accuracy |
| MILO          | ADEMAMIX      |  50.0444 |  48.9333 | MILO     | 0.210541    |               | final_validation_accuracy |
| MILO_LW       | SGD           |  48.8948 |  34.0207 | MILO_LW  | 0.000122418 | ***           | final_validation_accuracy |
| MILO_LW       | ADAMW         |  48.8948 |  48.0119 | MILO_LW  | 0.341182    |               | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  48.8948 |  14.963  | MILO_LW  | 1.09138e-13 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  48.8948 |  33.9467 | MILO_LW  | 4.94466e-08 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  48.8948 |  18.0504 | MILO_LW  | 8.2569e-11  | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  48.8948 |  48.9333 | ADEMAMIX | 0.961561    |               | final_validation_accuracy |
| SGD           | ADAMW         |  34.0207 |  48.0119 | ADAMW    | 1.02477e-05 | ***           | final_validation_accuracy |
| SGD           | ADAM_MINI     |  34.0207 |  14.963  | SGD      | 5.19206e-05 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  34.0207 |  33.9467 | SGD      | 0.950031    |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  34.0207 |  18.0504 | SGD      | 5.15771e-05 | ***           | final_validation_accuracy |
| SGD           | ADEMAMIX      |  34.0207 |  48.9333 | ADEMAMIX | 7.5214e-06  | ***           | final_validation_accuracy |
| ADAMW         | ADAM_MINI     |  48.0119 |  14.963  | ADAMW    | 1.62535e-06 | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  48.0119 |  33.9467 | ADAMW    | 9.15069e-06 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  48.0119 |  18.0504 | ADAMW    | 3.28449e-07 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  48.0119 |  48.9333 | ADEMAMIX | 0.425211    |               | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  14.963  |  33.9467 | NOVOGRAD | 1.24473e-07 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  14.963  |  18.0504 | ADAGRAD  | 0.00020771  | ***           | final_validation_accuracy |
| ADAM_MINI     | ADEMAMIX      |  14.963  |  48.9333 | ADEMAMIX | 9.30633e-07 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADAGRAD       |  33.9467 |  18.0504 | NOVOGRAD | 7.60713e-10 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADEMAMIX      |  33.9467 |  48.9333 | ADEMAMIX | 2.77494e-06 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  18.0504 |  48.9333 | ADEMAMIX | 1.13443e-07 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.4983 | 0.0054 | 0.0024 | 0.4916 | 0.5049 |
| MILO_LW | 0.4859 | 0.0052 | 0.0023 | 0.4794 | 0.4924 |
| SGD | 0.3347 | 0.0313 | 0.0140 | 0.2959 | 0.3735 |
| ADAMW | 0.4805 | 0.0161 | 0.0072 | 0.4605 | 0.5005 |
| ADAM_MINI | 0.1232 | 0.0045 | 0.0020 | 0.1176 | 0.1289 |
| NOVOGRAD | 0.3329 | 0.0082 | 0.0037 | 0.3227 | 0.3431 |
| ADAGRAD | 0.1524 | 0.0078 | 0.0035 | 0.1428 | 0.1620 |
| ADEMAMIX | 0.4880 | 0.0166 | 0.0074 | 0.4674 | 0.5085 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.498255 | 0.48591  | MILO     | 0.00612451  | **            | final_validation_f1_score |
| MILO          | SGD           | 0.498255 | 0.334707 | MILO     | 0.000233492 | ***           | final_validation_f1_score |
| MILO          | ADAMW         | 0.498255 | 0.480497 | MILO     | 0.0682005   |               | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.498255 | 0.123236 | MILO     | 5.62968e-14 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.498255 | 0.332941 | MILO     | 3.05562e-09 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.498255 | 0.152396 | MILO     | 7.36142e-12 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.498255 | 0.48796  | MILO     | 0.245151    |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.48591  | 0.334707 | MILO_LW  | 0.000327409 | ***           | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.48591  | 0.480497 | MILO_LW  | 0.508462    |               | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.48591  | 0.123236 | MILO_LW  | 4.99621e-14 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.48591  | 0.332941 | MILO_LW  | 6.4285e-09  | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.48591  | 0.152396 | MILO_LW  | 1.28528e-11 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.48591  | 0.48796  | ADEMAMIX | 0.802757    |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.334707 | 0.480497 | ADAMW    | 8.98138e-05 | ***           | final_validation_f1_score |
| SGD           | ADAM_MINI     | 0.334707 | 0.123236 | SGD      | 8.81348e-05 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.334707 | 0.332941 | SGD      | 0.907895    |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.334707 | 0.152396 | SGD      | 0.000110389 | ***           | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.334707 | 0.48796  | ADEMAMIX | 6.38298e-05 | ***           | final_validation_f1_score |
| ADAMW         | ADAM_MINI     | 0.480497 | 0.123236 | ADAMW    | 2.0892e-07  | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.480497 | 0.332941 | ADAMW    | 1.94121e-06 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.480497 | 0.152396 | ADAMW    | 2.57052e-08 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.480497 | 0.48796  | ADEMAMIX | 0.49111     |               | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.123236 | 0.332941 | NOVOGRAD | 2.36577e-09 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.123236 | 0.152396 | ADAGRAD  | 0.000248772 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADEMAMIX      | 0.123236 | 0.48796  | ADEMAMIX | 2.30147e-07 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADAGRAD       | 0.332941 | 0.152396 | NOVOGRAD | 4.32983e-10 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADEMAMIX      | 0.332941 | 0.48796  | ADEMAMIX | 1.88342e-06 | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.152396 | 0.48796  | ADEMAMIX | 3.08277e-08 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9555 | 0.0017 | 0.0007 | 0.9535 | 0.9576 |
| MILO_LW | 0.9539 | 0.0011 | 0.0005 | 0.9526 | 0.9553 |
| SGD | 0.9343 | 0.0042 | 0.0019 | 0.9290 | 0.9395 |
| ADAMW | 0.9668 | 0.0024 | 0.0011 | 0.9637 | 0.9698 |
| ADAM_MINI | 0.8419 | 0.0012 | 0.0005 | 0.8404 | 0.8433 |
| NOVOGRAD | 0.9306 | 0.0024 | 0.0011 | 0.9276 | 0.9335 |
| ADAGRAD | 0.8588 | 0.0051 | 0.0023 | 0.8524 | 0.8651 |
| ADEMAMIX | 0.9659 | 0.0024 | 0.0011 | 0.9630 | 0.9689 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.955543 | 0.95393  | MILO     | 0.112932    |               | final_validation_auc |
| MILO          | SGD           | 0.955543 | 0.934284 | MILO     | 0.000110515 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.955543 | 0.966761 | ADAMW    | 5.84645e-05 | ***           | final_validation_auc |
| MILO          | ADAM_MINI     | 0.955543 | 0.84188  | MILO     | 3.33118e-13 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.955543 | 0.930578 | MILO     | 2.01087e-07 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.955543 | 0.858759 | MILO     | 2.57801e-07 | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.955543 | 0.965942 | ADEMAMIX | 8.7685e-05  | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.95393  | 0.934284 | MILO_LW  | 0.000290019 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.95393  | 0.966761 | ADAMW    | 6.32593e-05 | ***           | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.95393  | 0.84188  | MILO_LW  | 3.19419e-15 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.95393  | 0.930578 | MILO_LW  | 1.95424e-06 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.95393  | 0.858759 | MILO_LW  | 8.13113e-07 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.95393  | 0.965942 | ADEMAMIX | 8.23881e-05 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.934284 | 0.966761 | ADAMW    | 3.45855e-06 | ***           | final_validation_auc |
| SGD           | ADAM_MINI     | 0.934284 | 0.84188  | SGD      | 2.45194e-07 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.934284 | 0.930578 | SGD      | 0.136815    |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.934284 | 0.858759 | SGD      | 9.33083e-09 | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.934284 | 0.965942 | ADEMAMIX | 4.22604e-06 | ***           | final_validation_auc |
| ADAMW         | ADAM_MINI     | 0.966761 | 0.84188  | ADAMW    | 1.4322e-10  | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.966761 | 0.930578 | ADAMW    | 1.05165e-08 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.966761 | 0.858759 | ADAMW    | 2.03354e-08 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.966761 | 0.965942 | ADAMW    | 0.607183    |               | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.84188  | 0.930578 | NOVOGRAD | 7.18706e-10 | ***           | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.84188  | 0.858759 | ADAGRAD  | 0.00131624  | **            | final_validation_auc |
| ADAM_MINI     | ADEMAMIX      | 0.84188  | 0.965942 | ADEMAMIX | 1.25031e-10 | ***           | final_validation_auc |
| NOVOGRAD      | ADAGRAD       | 0.930578 | 0.858759 | NOVOGRAD | 2.38396e-07 | ***           | final_validation_auc |
| NOVOGRAD      | ADEMAMIX      | 0.930578 | 0.965942 | ADEMAMIX | 1.19643e-08 | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.858759 | 0.965942 | ADEMAMIX | 2.29166e-08 | ***           | final_validation_auc |

