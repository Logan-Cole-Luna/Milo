# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 2.4756 | 0.0588 | 0.0263 | 2.4026 | 2.5487 |
| MILO_LW | 2.4534 | 0.0380 | 0.0170 | 2.4063 | 2.5006 |
| SGD | 2.8659 | 0.0326 | 0.0146 | 2.8254 | 2.9063 |
| ADAMW | 2.5234 | 0.0839 | 0.0375 | 2.4193 | 2.6275 |
| ADAM_MINI | 4.5441 | 0.0635 | 0.0284 | 4.4653 | 4.6229 |
| NOVOGRAD | 2.8547 | 0.0497 | 0.0222 | 2.7930 | 2.9165 |
| ADAGRAD | 3.0200 | 0.0401 | 0.0179 | 2.9702 | 3.0697 |
| ADEMAMIX | 2.9858 | 0.0374 | 0.0167 | 2.9393 | 3.0323 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  2.47562 |  2.45345 | MILO_LW  | 0.502568    |               | final_validation_loss |
| MILO          | SGD           |  2.47562 |  2.86588 | MILO     | 9.50382e-06 | ***           | final_validation_loss |
| MILO          | ADAMW         |  2.47562 |  2.52338 | MILO     | 0.330997    |               | final_validation_loss |
| MILO          | ADAM_MINI     |  2.47562 |  4.54406 | MILO     | 1.86572e-11 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      |  2.47562 |  2.85475 | MILO     | 5.11685e-06 | ***           | final_validation_loss |
| MILO          | ADAGRAD       |  2.47562 |  3.01998 | MILO     | 5.31353e-07 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      |  2.47562 |  2.98579 | MILO     | 1.06204e-06 | ***           | final_validation_loss |
| MILO_LW       | SGD           |  2.45345 |  2.86588 | MILO_LW  | 1.01287e-07 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         |  2.45345 |  2.52338 | MILO_LW  | 0.144077    |               | final_validation_loss |
| MILO_LW       | ADAM_MINI     |  2.45345 |  4.54406 | MILO_LW  | 2.30915e-10 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      |  2.45345 |  2.85475 | MILO_LW  | 1.02776e-06 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       |  2.45345 |  3.01998 | MILO_LW  | 1.43762e-08 | ***           | final_validation_loss |
| MILO_LW       | ADEMAMIX      |  2.45345 |  2.98579 | MILO_LW  | 1.72262e-08 | ***           | final_validation_loss |
| SGD           | ADAMW         |  2.86588 |  2.52338 | ADAMW    | 0.000308292 | ***           | final_validation_loss |
| SGD           | ADAM_MINI     |  2.86588 |  4.54406 | SGD      | 3.41092e-09 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      |  2.86588 |  2.85475 | NOVOGRAD | 0.688074    |               | final_validation_loss |
| SGD           | ADAGRAD       |  2.86588 |  3.01998 | SGD      | 0.000189031 | ***           | final_validation_loss |
| SGD           | ADEMAMIX      |  2.86588 |  2.98579 | SGD      | 0.000684686 | ***           | final_validation_loss |
| ADAMW         | ADAM_MINI     |  2.52338 |  4.54406 | ADAMW    | 3.36877e-10 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      |  2.52338 |  2.85475 | ADAMW    | 0.000182686 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       |  2.52338 |  3.01998 | ADAMW    | 2.86464e-05 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      |  2.52338 |  2.98579 | ADAMW    | 5.05903e-05 | ***           | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      |  4.54406 |  2.85475 | NOVOGRAD | 1.3403e-10  | ***           | final_validation_loss |
| ADAM_MINI     | ADAGRAD       |  4.54406 |  3.01998 | ADAGRAD  | 1.19554e-09 | ***           | final_validation_loss |
| ADAM_MINI     | ADEMAMIX      |  4.54406 |  2.98579 | ADEMAMIX | 1.78258e-09 | ***           | final_validation_loss |
| NOVOGRAD      | ADAGRAD       |  2.85475 |  3.01998 | NOVOGRAD | 0.000483865 | ***           | final_validation_loss |
| NOVOGRAD      | ADEMAMIX      |  2.85475 |  2.98579 | NOVOGRAD | 0.00185626  | **            | final_validation_loss |
| ADAGRAD       | ADEMAMIX      |  3.01998 |  2.98579 | ADEMAMIX | 0.200865    |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 39.6741 | 1.0879 | 0.4865 | 38.3233 | 41.0249 |
| MILO_LW | 38.3852 | 1.1056 | 0.4944 | 37.0124 | 39.7579 |
| SGD | 31.4252 | 0.6084 | 0.2721 | 30.6697 | 32.1807 |
| ADAMW | 33.4400 | 1.6234 | 0.7260 | 31.4243 | 35.4557 |
| ADAM_MINI | 1.4993 | 0.6663 | 0.2980 | 0.6720 | 2.3266 |
| NOVOGRAD | 32.9215 | 0.5844 | 0.2614 | 32.1958 | 33.6471 |
| ADAGRAD | 32.6519 | 0.5922 | 0.2648 | 31.9165 | 33.3872 |
| ADEMAMIX | 21.7778 | 1.0640 | 0.4758 | 20.4567 | 23.0989 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 39.6741  | 38.3852  | MILO     | 0.100232    |               | final_validation_accuracy |
| MILO          | SGD           | 39.6741  | 31.4252  | MILO     | 4.06477e-06 | ***           | final_validation_accuracy |
| MILO          | ADAMW         | 39.6741  | 33.44    | MILO     | 0.000189415 | ***           | final_validation_accuracy |
| MILO          | ADAM_MINI     | 39.6741  |  1.49926 | MILO     | 1.23715e-10 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      | 39.6741  | 32.9215  | MILO     | 1.55374e-05 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       | 39.6741  | 32.6519  | MILO     | 1.18142e-05 | ***           | final_validation_accuracy |
| MILO          | ADEMAMIX      | 39.6741  | 21.7778  | MILO     | 4.73148e-09 | ***           | final_validation_accuracy |
| MILO_LW       | SGD           | 38.3852  | 31.4252  | MILO_LW  | 1.32652e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADAMW         | 38.3852  | 33.44    | MILO_LW  | 0.000770053 | ***           | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     | 38.3852  |  1.49926 | MILO_LW  | 2.00437e-10 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      | 38.3852  | 32.9215  | MILO_LW  | 6.13946e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       | 38.3852  | 32.6519  | MILO_LW  | 4.50043e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      | 38.3852  | 21.7778  | MILO_LW  | 9.24584e-09 | ***           | final_validation_accuracy |
| SGD           | ADAMW         | 31.4252  | 33.44    | ADAMW    | 0.0474024   | *             | final_validation_accuracy |
| SGD           | ADAM_MINI     | 31.4252  |  1.49926 | SGD      | 1.4637e-12  | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      | 31.4252  | 32.9215  | NOVOGRAD | 0.00415623  | **            | final_validation_accuracy |
| SGD           | ADAGRAD       | 31.4252  | 32.6519  | ADAGRAD  | 0.01206     | *             | final_validation_accuracy |
| SGD           | ADEMAMIX      | 31.4252  | 21.7778  | SGD      | 1.22954e-06 | ***           | final_validation_accuracy |
| ADAMW         | ADAM_MINI     | 33.44    |  1.49926 | ADAMW    | 7.82684e-08 | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      | 33.44    | 32.9215  | ADAMW    | 0.531275    |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       | 33.44    | 32.6519  | ADAMW    | 0.354179    |               | final_validation_accuracy |
| ADAMW         | ADEMAMIX      | 33.44    | 21.7778  | ADAMW    | 3.35341e-06 | ***           | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  1.49926 | 32.9215  | NOVOGRAD | 1.05249e-12 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  1.49926 | 32.6519  | ADAGRAD  | 1.09647e-12 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADEMAMIX      |  1.49926 | 21.7778  | ADEMAMIX | 6.003e-09   | ***           | final_validation_accuracy |
| NOVOGRAD      | ADAGRAD       | 32.9215  | 32.6519  | NOVOGRAD | 0.489335    |               | final_validation_accuracy |
| NOVOGRAD      | ADEMAMIX      | 32.9215  | 21.7778  | NOVOGRAD | 6.05362e-07 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      | 32.6519  | 21.7778  | ADAGRAD  | 6.60554e-07 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.3911 | 0.0120 | 0.0053 | 0.3762 | 0.4059 |
| MILO_LW | 0.3777 | 0.0112 | 0.0050 | 0.3638 | 0.3916 |
| SGD | 0.3007 | 0.0073 | 0.0033 | 0.2916 | 0.3098 |
| ADAMW | 0.3201 | 0.0169 | 0.0076 | 0.2992 | 0.3411 |
| ADAM_MINI | 0.0025 | 0.0026 | 0.0012 | -0.0007 | 0.0058 |
| NOVOGRAD | 0.3184 | 0.0049 | 0.0022 | 0.3122 | 0.3245 |
| ADAGRAD | 0.3240 | 0.0059 | 0.0026 | 0.3167 | 0.3313 |
| ADEMAMIX | 0.1817 | 0.0111 | 0.0049 | 0.1680 | 0.1955 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |     Mean A |     Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|-----------:|-----------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.391052   | 0.377723   | MILO     | 0.106471    |               | final_validation_f1_score |
| MILO          | SGD           | 0.391052   | 0.300689   | MILO     | 3.02367e-06 | ***           | final_validation_f1_score |
| MILO          | ADAMW         | 0.391052   | 0.320138   | MILO     | 0.000103178 | ***           | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.391052   | 0.00252438 | MILO     | 7.13649e-08 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.391052   | 0.318352   | MILO     | 3.68844e-05 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.391052   | 0.323963   | MILO     | 3.59549e-05 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.391052   | 0.181726   | MILO     | 2.5516e-09  | ***           | final_validation_f1_score |
| MILO_LW       | SGD           | 0.377723   | 0.300689   | MILO_LW  | 4.50051e-06 | ***           | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.377723   | 0.320138   | MILO_LW  | 0.000395407 | ***           | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.377723   | 0.00252438 | MILO_LW  | 5.31665e-08 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.377723   | 0.318352   | MILO_LW  | 6.37992e-05 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.377723   | 0.323963   | MILO_LW  | 7.32903e-05 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.377723   | 0.181726   | MILO_LW  | 2.98122e-09 | ***           | final_validation_f1_score |
| SGD           | ADAMW         | 0.300689   | 0.320138   | ADAMW    | 0.0602705   |               | final_validation_f1_score |
| SGD           | ADAM_MINI     | 0.300689   | 0.00252438 | SGD      | 3.9946e-09  | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.300689   | 0.318352   | NOVOGRAD | 0.00284521  | **            | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.300689   | 0.323963   | ADAGRAD  | 0.000632386 | ***           | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.300689   | 0.181726   | SGD      | 2.12475e-07 | ***           | final_validation_f1_score |
| ADAMW         | ADAM_MINI     | 0.320138   | 0.00252438 | ADAMW    | 1.21172e-06 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.320138   | 0.318352   | ADAMW    | 0.830041    |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.320138   | 0.323963   | ADAGRAD  | 0.65272     |               | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.320138   | 0.181726   | ADAMW    | 1.38914e-06 | ***           | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.00252438 | 0.318352   | NOVOGRAD | 1.27573e-11 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.00252438 | 0.323963   | ADAGRAD  | 1.79525e-10 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADEMAMIX      | 0.00252438 | 0.181726   | ADEMAMIX | 1.29966e-06 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADAGRAD       | 0.318352   | 0.323963   | ADAGRAD  | 0.141829    |               | final_validation_f1_score |
| NOVOGRAD      | ADEMAMIX      | 0.318352   | 0.181726   | NOVOGRAD | 6.31577e-07 | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.323963   | 0.181726   | ADAGRAD  | 2.05709e-07 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9405 | 0.0027 | 0.0012 | 0.9371 | 0.9438 |
| MILO_LW | 0.9418 | 0.0018 | 0.0008 | 0.9396 | 0.9441 |
| SGD | 0.9290 | 0.0016 | 0.0007 | 0.9270 | 0.9311 |
| ADAMW | 0.9433 | 0.0034 | 0.0015 | 0.9390 | 0.9475 |
| ADAM_MINI | 0.6105 | 0.0713 | 0.0319 | 0.5220 | 0.6991 |
| NOVOGRAD | 0.9292 | 0.0022 | 0.0010 | 0.9265 | 0.9319 |
| ADAGRAD | 0.9214 | 0.0022 | 0.0010 | 0.9186 | 0.9242 |
| ADEMAMIX | 0.9131 | 0.0020 | 0.0009 | 0.9106 | 0.9156 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.940489 | 0.941849 | MILO_LW  | 0.380723    |               | final_validation_auc |
| MILO          | SGD           | 0.940489 | 0.929045 | MILO     | 0.000116093 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.940489 | 0.943252 | ADAMW    | 0.19745     |               | final_validation_auc |
| MILO          | ADAM_MINI     | 0.940489 | 0.610526 | MILO     | 0.000486741 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.940489 | 0.929204 | MILO     | 0.000107274 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.940489 | 0.921437 | MILO     | 2.5964e-06  | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.940489 | 0.913091 | MILO     | 2.03499e-07 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.941849 | 0.929045 | MILO_LW  | 2.81239e-06 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.941849 | 0.943252 | ADAMW    | 0.450595    |               | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.941849 | 0.610526 | MILO_LW  | 0.000482159 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.941849 | 0.929204 | MILO_LW  | 1.1118e-05  | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.941849 | 0.921437 | MILO_LW  | 4.06019e-07 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.941849 | 0.913091 | MILO_LW  | 1.27263e-08 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.929045 | 0.943252 | ADAMW    | 0.00020794  | ***           | final_validation_auc |
| SGD           | ADAM_MINI     | 0.929045 | 0.610526 | SGD      | 0.000562406 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.929045 | 0.929204 | NOVOGRAD | 0.899876    |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.929045 | 0.921437 | SGD      | 0.000408735 | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.929045 | 0.913091 | SGD      | 1.154e-06   | ***           | final_validation_auc |
| ADAMW         | ADAM_MINI     | 0.943252 | 0.610526 | ADAMW    | 0.000467667 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.943252 | 0.929204 | ADAMW    | 0.00013697  | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.943252 | 0.921437 | ADAMW    | 7.76488e-06 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.943252 | 0.913091 | ADAMW    | 1.35538e-06 | ***           | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.610526 | 0.929204 | NOVOGRAD | 0.000559446 | ***           | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.610526 | 0.921437 | ADAGRAD  | 0.000615305 | ***           | final_validation_auc |
| ADAM_MINI     | ADEMAMIX      | 0.610526 | 0.913091 | ADEMAMIX | 0.000684609 | ***           | final_validation_auc |
| NOVOGRAD      | ADAGRAD       | 0.929204 | 0.921437 | NOVOGRAD | 0.000551097 | ***           | final_validation_auc |
| NOVOGRAD      | ADEMAMIX      | 0.929204 | 0.913091 | NOVOGRAD | 2.12234e-06 | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.921437 | 0.913091 | ADAGRAD  | 0.000281332 | ***           | final_validation_auc |

