# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 2.3047 | 0.0000 | 0.0000 | 2.3047 | 2.3047 |
| MILO_LW | 2.3700 | 0.0000 | 0.0000 | 2.3700 | 2.3700 |
| SGD | 2.6354 | 0.0000 | 0.0000 | 2.6354 | 2.6354 |
| ADAMW | 2.3030 | 0.0000 | 0.0000 | 2.3030 | 2.3030 |
| ADAGRAD | 3.4215 | 0.0000 | 0.0000 | 3.4215 | 3.4215 |
| ADEMAMIX | 2.1551 | 0.0000 | 0.0000 | 2.1551 | 2.1551 |
| SOAP | 1.9154 | 0.0000 | 0.0000 | 1.9154 | 1.9154 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  2.3047  |  2.37003 | MILO     |       nan |               | final_validation_loss |
| MILO          | SGD           |  2.3047  |  2.63539 | MILO     |       nan |               | final_validation_loss |
| MILO          | ADAMW         |  2.3047  |  2.30302 | ADAMW    |       nan |               | final_validation_loss |
| MILO          | ADAGRAD       |  2.3047  |  3.4215  | MILO     |       nan |               | final_validation_loss |
| MILO          | ADEMAMIX      |  2.3047  |  2.15509 | ADEMAMIX |       nan |               | final_validation_loss |
| MILO          | SOAP          |  2.3047  |  1.91542 | SOAP     |       nan |               | final_validation_loss |
| MILO_LW       | SGD           |  2.37003 |  2.63539 | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADAMW         |  2.37003 |  2.30302 | ADAMW    |       nan |               | final_validation_loss |
| MILO_LW       | ADAGRAD       |  2.37003 |  3.4215  | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADEMAMIX      |  2.37003 |  2.15509 | ADEMAMIX |       nan |               | final_validation_loss |
| MILO_LW       | SOAP          |  2.37003 |  1.91542 | SOAP     |       nan |               | final_validation_loss |
| SGD           | ADAMW         |  2.63539 |  2.30302 | ADAMW    |       nan |               | final_validation_loss |
| SGD           | ADAGRAD       |  2.63539 |  3.4215  | SGD      |       nan |               | final_validation_loss |
| SGD           | ADEMAMIX      |  2.63539 |  2.15509 | ADEMAMIX |       nan |               | final_validation_loss |
| SGD           | SOAP          |  2.63539 |  1.91542 | SOAP     |       nan |               | final_validation_loss |
| ADAMW         | ADAGRAD       |  2.30302 |  3.4215  | ADAMW    |       nan |               | final_validation_loss |
| ADAMW         | ADEMAMIX      |  2.30302 |  2.15509 | ADEMAMIX |       nan |               | final_validation_loss |
| ADAMW         | SOAP          |  2.30302 |  1.91542 | SOAP     |       nan |               | final_validation_loss |
| ADAGRAD       | ADEMAMIX      |  3.4215  |  2.15509 | ADEMAMIX |       nan |               | final_validation_loss |
| ADAGRAD       | SOAP          |  3.4215  |  1.91542 | SOAP     |       nan |               | final_validation_loss |
| ADEMAMIX      | SOAP          |  2.15509 |  1.91542 | SOAP     |       nan |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 41.0963 | 0.0000 | 0.0000 | 41.0963 | 41.0963 |
| MILO_LW | 39.4222 | 0.0000 | 0.0000 | 39.4222 | 39.4222 |
| SGD | 32.1037 | 0.0000 | 0.0000 | 32.1037 | 32.1037 |
| ADAMW | 39.2444 | 0.0000 | 0.0000 | 39.2444 | 39.2444 |
| ADAGRAD | 18.4296 | 0.0000 | 0.0000 | 18.4296 | 18.4296 |
| ADEMAMIX | 43.8074 | 0.0000 | 0.0000 | 43.8074 | 43.8074 |
| SOAP | 55.3778 | 0.0000 | 0.0000 | 55.3778 | 55.3778 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  41.0963 |  39.4222 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | SGD           |  41.0963 |  32.1037 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAMW         |  41.0963 |  39.2444 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAGRAD       |  41.0963 |  18.4296 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADEMAMIX      |  41.0963 |  43.8074 | ADEMAMIX |       nan |               | final_validation_accuracy |
| MILO          | SOAP          |  41.0963 |  55.3778 | SOAP     |       nan |               | final_validation_accuracy |
| MILO_LW       | SGD           |  39.4222 |  32.1037 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  39.4222 |  39.2444 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  39.4222 |  18.4296 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  39.4222 |  43.8074 | ADEMAMIX |       nan |               | final_validation_accuracy |
| MILO_LW       | SOAP          |  39.4222 |  55.3778 | SOAP     |       nan |               | final_validation_accuracy |
| SGD           | ADAMW         |  32.1037 |  39.2444 | ADAMW    |       nan |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  32.1037 |  18.4296 | SGD      |       nan |               | final_validation_accuracy |
| SGD           | ADEMAMIX      |  32.1037 |  43.8074 | ADEMAMIX |       nan |               | final_validation_accuracy |
| SGD           | SOAP          |  32.1037 |  55.3778 | SOAP     |       nan |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  39.2444 |  18.4296 | ADAMW    |       nan |               | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  39.2444 |  43.8074 | ADEMAMIX |       nan |               | final_validation_accuracy |
| ADAMW         | SOAP          |  39.2444 |  55.3778 | SOAP     |       nan |               | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  18.4296 |  43.8074 | ADEMAMIX |       nan |               | final_validation_accuracy |
| ADAGRAD       | SOAP          |  18.4296 |  55.3778 | SOAP     |       nan |               | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  43.8074 |  55.3778 | SOAP     |       nan |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.4042 | 0.0000 | 0.0000 | 0.4042 | 0.4042 |
| MILO_LW | 0.3857 | 0.0000 | 0.0000 | 0.3857 | 0.3857 |
| SGD | 0.3072 | 0.0000 | 0.0000 | 0.3072 | 0.3072 |
| ADAMW | 0.3865 | 0.0000 | 0.0000 | 0.3865 | 0.3865 |
| ADAGRAD | 0.1610 | 0.0000 | 0.0000 | 0.1610 | 0.1610 |
| ADEMAMIX | 0.4278 | 0.0000 | 0.0000 | 0.4278 | 0.4278 |
| SOAP | 0.5514 | 0.0000 | 0.0000 | 0.5514 | 0.5514 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.404217 | 0.385669 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | SGD           | 0.404217 | 0.307182 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.404217 | 0.386456 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.404217 | 0.160975 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.404217 | 0.427794 | ADEMAMIX |       nan |               | final_validation_f1_score |
| MILO          | SOAP          | 0.404217 | 0.551441 | SOAP     |       nan |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.385669 | 0.307182 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.385669 | 0.386456 | ADAMW    |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.385669 | 0.160975 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.385669 | 0.427794 | ADEMAMIX |       nan |               | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.385669 | 0.551441 | SOAP     |       nan |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.307182 | 0.386456 | ADAMW    |       nan |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.307182 | 0.160975 | SGD      |       nan |               | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.307182 | 0.427794 | ADEMAMIX |       nan |               | final_validation_f1_score |
| SGD           | SOAP          | 0.307182 | 0.551441 | SOAP     |       nan |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.386456 | 0.160975 | ADAMW    |       nan |               | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.386456 | 0.427794 | ADEMAMIX |       nan |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.386456 | 0.551441 | SOAP     |       nan |               | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.160975 | 0.427794 | ADEMAMIX |       nan |               | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.160975 | 0.551441 | SOAP     |       nan |               | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.427794 | 0.551441 | SOAP     |       nan |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9494 | 0.0000 | 0.0000 | 0.9494 | 0.9494 |
| MILO_LW | 0.9462 | 0.0000 | 0.0000 | 0.9462 | 0.9462 |
| SGD | 0.9338 | 0.0000 | 0.0000 | 0.9338 | 0.9338 |
| ADAMW | 0.9555 | 0.0000 | 0.0000 | 0.9555 | 0.9555 |
| ADAGRAD | 0.8637 | 0.0000 | 0.0000 | 0.8637 | 0.8637 |
| ADEMAMIX | 0.9614 | 0.0000 | 0.0000 | 0.9614 | 0.9614 |
| SOAP | 0.9735 | 0.0000 | 0.0000 | 0.9735 | 0.9735 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.949373 | 0.946185 | MILO     |       nan |               | final_validation_auc |
| MILO          | SGD           | 0.949373 | 0.93382  | MILO     |       nan |               | final_validation_auc |
| MILO          | ADAMW         | 0.949373 | 0.955457 | ADAMW    |       nan |               | final_validation_auc |
| MILO          | ADAGRAD       | 0.949373 | 0.863687 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADEMAMIX      | 0.949373 | 0.961403 | ADEMAMIX |       nan |               | final_validation_auc |
| MILO          | SOAP          | 0.949373 | 0.97352  | SOAP     |       nan |               | final_validation_auc |
| MILO_LW       | SGD           | 0.946185 | 0.93382  | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.946185 | 0.955457 | ADAMW    |       nan |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.946185 | 0.863687 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.946185 | 0.961403 | ADEMAMIX |       nan |               | final_validation_auc |
| MILO_LW       | SOAP          | 0.946185 | 0.97352  | SOAP     |       nan |               | final_validation_auc |
| SGD           | ADAMW         | 0.93382  | 0.955457 | ADAMW    |       nan |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.93382  | 0.863687 | SGD      |       nan |               | final_validation_auc |
| SGD           | ADEMAMIX      | 0.93382  | 0.961403 | ADEMAMIX |       nan |               | final_validation_auc |
| SGD           | SOAP          | 0.93382  | 0.97352  | SOAP     |       nan |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.955457 | 0.863687 | ADAMW    |       nan |               | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.955457 | 0.961403 | ADEMAMIX |       nan |               | final_validation_auc |
| ADAMW         | SOAP          | 0.955457 | 0.97352  | SOAP     |       nan |               | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.863687 | 0.961403 | ADEMAMIX |       nan |               | final_validation_auc |
| ADAGRAD       | SOAP          | 0.863687 | 0.97352  | SOAP     |       nan |               | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.961403 | 0.97352  | SOAP     |       nan |               | final_validation_auc |

