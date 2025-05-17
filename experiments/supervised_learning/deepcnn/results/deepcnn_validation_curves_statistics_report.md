# Statistical Analysis Report

## Methodology

Standard deviations, standard errors, and 95% confidence intervals were computed using the t-distribution.

### Validation_Loss Statistics

Number of runs: 5

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo | 0.7943 | 0.0334 | 0.0150 | 0.7528 | 0.8358 |
| milo_LW | 0.6426 | 0.0132 | 0.0059 | 0.6262 | 0.6590 |
| SGD | 0.6192 | 0.0312 | 0.0140 | 0.5804 | 0.6580 |
| ADAGRAD | 1.0939 | 0.0075 | 0.0034 | 1.0845 | 1.1032 |
| ADAMW | 0.6627 | 0.0371 | 0.0166 | 0.6166 | 0.7088 |
| NOVOGRAD | 0.6122 | 0.0383 | 0.0171 | 0.5647 | 0.6597 |

#### Pairwise Significance Tests

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| milo           | milo_LW        | 0.794286 | 0.6426   | milo_LW   | 0.000179402 | ***           | final_validation_loss |
| milo           | SGD           | 0.794286 | 0.619184 | SGD      | 2.75675e-05 | ***           | final_validation_loss |
| milo           | ADAGRAD       | 0.794286 | 1.09386  | milo      | 1.89083e-05 | ***           | final_validation_loss |
| milo           | ADAMW         | 0.794286 | 0.662692 | ADAMW    | 0.000381691 | ***           | final_validation_loss |
| milo           | NOVOGRAD      | 0.794286 | 0.612229 | NOVOGRAD | 4.77891e-05 | ***           | final_validation_loss |
| milo_LW        | SGD           | 0.6426   | 0.619184 | SGD      | 0.179       |               | final_validation_loss |
| milo_LW        | ADAGRAD       | 0.6426   | 1.09386  | milo_LW   | 2.74223e-10 | ***           | final_validation_loss |
| milo_LW        | ADAMW         | 0.6426   | 0.662692 | milo_LW   | 0.305877    |               | final_validation_loss |
| milo_LW        | NOVOGRAD      | 0.6426   | 0.612229 | NOVOGRAD | 0.154933    |               | final_validation_loss |
| SGD           | ADAGRAD       | 0.619184 | 1.09386  | SGD      | 1.64973e-06 | ***           | final_validation_loss |
| SGD           | ADAMW         | 0.619184 | 0.662692 | SGD      | 0.0809086   |               | final_validation_loss |
| SGD           | NOVOGRAD      | 0.619184 | 0.612229 | NOVOGRAD | 0.761208    |               | final_validation_loss |
| ADAGRAD       | ADAMW         | 1.09386  | 0.662692 | ADAMW    | 7.0015e-06  | ***           | final_validation_loss |
| ADAGRAD       | NOVOGRAD      | 1.09386  | 0.612229 | NOVOGRAD | 5.13574e-06 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.662692 | 0.612229 | NOVOGRAD | 0.0672261   |               | final_validation_loss |

### Validation_Accuracy Statistics

Number of runs: 5

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo | 79.4133 | 0.2927 | 0.1309 | 79.0499 | 79.7767 |
| milo_LW | 79.1763 | 0.4454 | 0.1992 | 78.6233 | 79.7293 |
| SGD | 81.3985 | 0.7278 | 0.3255 | 80.4949 | 82.3022 |
| ADAGRAD | 60.8237 | 0.4960 | 0.2218 | 60.2079 | 61.4395 |
| ADAMW | 80.6163 | 0.7723 | 0.3454 | 79.6573 | 81.5753 |
| NOVOGRAD | 80.6993 | 0.5096 | 0.2279 | 80.0665 | 81.3320 |

#### Pairwise Significance Tests

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| milo           | milo_LW        |  79.4133 |  79.1763 | milo      | 0.353496    |               | final_validation_accuracy |
| milo           | SGD           |  79.4133 |  81.3985 | SGD      | 0.00202953  | **            | final_validation_accuracy |
| milo           | ADAGRAD       |  79.4133 |  60.8237 | milo      | 1.14582e-10 | ***           | final_validation_accuracy |
| milo           | ADAMW         |  79.4133 |  80.6163 | ADAMW    | 0.0217283   | *             | final_validation_accuracy |
| milo           | NOVOGRAD      |  79.4133 |  80.6993 | NOVOGRAD | 0.00229857  | **            | final_validation_accuracy |
| milo_LW        | SGD           |  79.1763 |  81.3985 | SGD      | 0.000790293 | ***           | final_validation_accuracy |
| milo_LW        | ADAGRAD       |  79.1763 |  60.8237 | milo_LW   | 6.85052e-12 | ***           | final_validation_accuracy |
| milo_LW        | ADAMW         |  79.1763 |  80.6163 | ADAMW    | 0.0100438   | *             | final_validation_accuracy |
| milo_LW        | NOVOGRAD      |  79.1763 |  80.6993 | NOVOGRAD | 0.00106774  | **            | final_validation_accuracy |
| SGD           | ADAGRAD       |  81.3985 |  60.8237 | SGD      | 2.13885e-10 | ***           | final_validation_accuracy |
| SGD           | ADAMW         |  81.3985 |  80.6163 | SGD      | 0.138054    |               | final_validation_accuracy |
| SGD           | NOVOGRAD      |  81.3985 |  80.6993 | SGD      | 0.120853    |               | final_validation_accuracy |
| ADAGRAD       | ADAMW         |  60.8237 |  80.6163 | ADAMW    | 6.75306e-10 | ***           | final_validation_accuracy |
| ADAGRAD       | NOVOGRAD      |  60.8237 |  80.6993 | NOVOGRAD | 4.8512e-12  | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  80.6163 |  80.6993 | NOVOGRAD | 0.846859    |               | final_validation_accuracy |

### Validation_F1_Score Statistics

Number of runs: 5

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo | 0.7933 | 0.0029 | 0.0013 | 0.7897 | 0.7969 |
| milo_LW | 0.7912 | 0.0056 | 0.0025 | 0.7843 | 0.7981 |
| SGD | 0.8131 | 0.0076 | 0.0034 | 0.8037 | 0.8225 |
| ADAGRAD | 0.6047 | 0.0055 | 0.0025 | 0.5978 | 0.6115 |
| ADAMW | 0.8050 | 0.0079 | 0.0035 | 0.7952 | 0.8149 |
| NOVOGRAD | 0.8061 | 0.0050 | 0.0022 | 0.7999 | 0.8123 |

#### Pairwise Significance Tests

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| milo           | milo_LW        | 0.793328 | 0.791208 | milo      | 0.47736     |               | final_validation_f1_score |
| milo           | SGD           | 0.793328 | 0.813113 | SGD      | 0.00260751  | **            | final_validation_f1_score |
| milo           | ADAGRAD       | 0.793328 | 0.604681 | milo      | 6.15235e-10 | ***           | final_validation_f1_score |
| milo           | ADAMW         | 0.793328 | 0.805031 | ADAMW    | 0.0263444   | *             | final_validation_f1_score |
| milo           | NOVOGRAD      | 0.793328 | 0.806104 | NOVOGRAD | 0.00210404  | **            | final_validation_f1_score |
| milo_LW        | SGD           | 0.791208 | 0.813113 | SGD      | 0.00107395  | **            | final_validation_f1_score |
| milo_LW        | ADAGRAD       | 0.791208 | 0.604681 | milo_LW   | 1.69928e-11 | ***           | final_validation_f1_score |
| milo_LW        | ADAMW         | 0.791208 | 0.805031 | ADAMW    | 0.0146505   | *             | final_validation_f1_score |
| milo_LW        | NOVOGRAD      | 0.791208 | 0.806104 | NOVOGRAD | 0.00215875  | **            | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.813113 | 0.604681 | SGD      | 1.65743e-10 | ***           | final_validation_f1_score |
| SGD           | ADAMW         | 0.813113 | 0.805031 | SGD      | 0.13799     |               | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.813113 | 0.806104 | SGD      | 0.128465    |               | final_validation_f1_score |
| ADAGRAD       | ADAMW         | 0.604681 | 0.805031 | ADAMW    | 3.994e-10   | ***           | final_validation_f1_score |
| ADAGRAD       | NOVOGRAD      | 0.604681 | 0.806104 | NOVOGRAD | 7.49107e-12 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.805031 | 0.806104 | NOVOGRAD | 0.805132    |               | final_validation_f1_score |

### Validation_Auc Statistics

Number of runs: 5

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo | 0.9759 | 0.0005 | 0.0002 | 0.9753 | 0.9764 |
| milo_LW | 0.9757 | 0.0011 | 0.0005 | 0.9743 | 0.9770 |
| SGD | 0.9807 | 0.0011 | 0.0005 | 0.9794 | 0.9821 |
| ADAGRAD | 0.9287 | 0.0009 | 0.0004 | 0.9276 | 0.9299 |
| ADAMW | 0.9787 | 0.0012 | 0.0006 | 0.9772 | 0.9802 |
| NOVOGRAD | 0.9799 | 0.0012 | 0.0005 | 0.9784 | 0.9814 |

#### Pairwise Significance Tests

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| milo           | milo_LW        | 0.975879 | 0.975668 | milo      | 0.703804    |               | final_validation_auc |
| milo           | SGD           | 0.975879 | 0.980739 | SGD      | 0.000151702 | ***           | final_validation_auc |
| milo           | ADAGRAD       | 0.975879 | 0.928724 | milo      | 1.45138e-10 | ***           | final_validation_auc |
| milo           | ADAMW         | 0.975879 | 0.978684 | ADAMW    | 0.00481965  | **            | final_validation_auc |
| milo           | NOVOGRAD      | 0.975879 | 0.979911 | NOVOGRAD | 0.000834993 | ***           | final_validation_auc |
| milo_LW        | SGD           | 0.975668 | 0.980739 | SGD      | 7.2133e-05  | ***           | final_validation_auc |
| milo_LW        | ADAGRAD       | 0.975668 | 0.928724 | milo_LW   | 2.09822e-12 | ***           | final_validation_auc |
| milo_LW        | ADAMW         | 0.975668 | 0.978684 | ADAMW    | 0.00351263  | **            | final_validation_auc |
| milo_LW        | NOVOGRAD      | 0.975668 | 0.979911 | NOVOGRAD | 0.000397222 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.980739 | 0.928724 | SGD      | 7.88312e-13 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.980739 | 0.978684 | SGD      | 0.0229611   | *             | final_validation_auc |
| SGD           | NOVOGRAD      | 0.980739 | 0.979911 | SGD      | 0.28313     |               | final_validation_auc |
| ADAGRAD       | ADAMW         | 0.928724 | 0.978684 | ADAMW    | 6.50376e-12 | ***           | final_validation_auc |
| ADAGRAD       | NOVOGRAD      | 0.928724 | 0.979911 | NOVOGRAD | 3.92426e-12 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.978684 | 0.979911 | NOVOGRAD | 0.150256    |               | final_validation_auc |

