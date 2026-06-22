# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 2.6641 | 0.0000 | 0.0000 | 2.6641 | 2.6641 |
| MILO_LW | 2.6969 | 0.0000 | 0.0000 | 2.6969 | 2.6969 |
| SGD | 2.9642 | 0.0000 | 0.0000 | 2.9642 | 2.9642 |
| ADAMW | 3.0990 | 0.0000 | 0.0000 | 3.0990 | 3.0990 |
| ADAGRAD | 2.8202 | 0.0000 | 0.0000 | 2.8202 | 2.8202 |
| ADEMAMIX | 3.5873 | 0.0000 | 0.0000 | 3.5873 | 3.5873 |
| SOAP | nan | 0.0000 | 0.0000 | nan | nan |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |    Mean B | Better   |   p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|----------:|:---------|----------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  2.66411 |   2.69685 | MILO     |       nan |               | final_validation_loss |
| MILO          | SGD           |  2.66411 |   2.96416 | MILO     |       nan |               | final_validation_loss |
| MILO          | ADAMW         |  2.66411 |   3.09903 | MILO     |       nan |               | final_validation_loss |
| MILO          | ADAGRAD       |  2.66411 |   2.8202  | MILO     |       nan |               | final_validation_loss |
| MILO          | ADEMAMIX      |  2.66411 |   3.58725 | MILO     |       nan |               | final_validation_loss |
| MILO          | SOAP          |  2.66411 | nan       | SOAP     |       nan |               | final_validation_loss |
| MILO_LW       | SGD           |  2.69685 |   2.96416 | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADAMW         |  2.69685 |   3.09903 | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADAGRAD       |  2.69685 |   2.8202  | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADEMAMIX      |  2.69685 |   3.58725 | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | SOAP          |  2.69685 | nan       | SOAP     |       nan |               | final_validation_loss |
| SGD           | ADAMW         |  2.96416 |   3.09903 | SGD      |       nan |               | final_validation_loss |
| SGD           | ADAGRAD       |  2.96416 |   2.8202  | ADAGRAD  |       nan |               | final_validation_loss |
| SGD           | ADEMAMIX      |  2.96416 |   3.58725 | SGD      |       nan |               | final_validation_loss |
| SGD           | SOAP          |  2.96416 | nan       | SOAP     |       nan |               | final_validation_loss |
| ADAMW         | ADAGRAD       |  3.09903 |   2.8202  | ADAGRAD  |       nan |               | final_validation_loss |
| ADAMW         | ADEMAMIX      |  3.09903 |   3.58725 | ADAMW    |       nan |               | final_validation_loss |
| ADAMW         | SOAP          |  3.09903 | nan       | SOAP     |       nan |               | final_validation_loss |
| ADAGRAD       | ADEMAMIX      |  2.8202  |   3.58725 | ADAGRAD  |       nan |               | final_validation_loss |
| ADAGRAD       | SOAP          |  2.8202  | nan       | SOAP     |       nan |               | final_validation_loss |
| ADEMAMIX      | SOAP          |  3.58725 | nan       | SOAP     |       nan |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 31.6296 | 0.0000 | 0.0000 | 31.6296 | 31.6296 |
| MILO_LW | 30.9037 | 0.0000 | 0.0000 | 30.9037 | 30.9037 |
| SGD | 24.6370 | 0.0000 | 0.0000 | 24.6370 | 24.6370 |
| ADAMW | 19.2889 | 0.0000 | 0.0000 | 19.2889 | 19.2889 |
| ADAGRAD | 30.0444 | 0.0000 | 0.0000 | 30.0444 | 30.0444 |
| ADEMAMIX | 10.3407 | 0.0000 | 0.0000 | 10.3407 | 10.3407 |
| SOAP | 1.0667 | 0.0000 | 0.0000 | 1.0667 | 1.0667 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  31.6296 | 30.9037  | MILO     |       nan |               | final_validation_accuracy |
| MILO          | SGD           |  31.6296 | 24.637   | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAMW         |  31.6296 | 19.2889  | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAGRAD       |  31.6296 | 30.0444  | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADEMAMIX      |  31.6296 | 10.3407  | MILO     |       nan |               | final_validation_accuracy |
| MILO          | SOAP          |  31.6296 |  1.06667 | MILO     |       nan |               | final_validation_accuracy |
| MILO_LW       | SGD           |  30.9037 | 24.637   | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  30.9037 | 19.2889  | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  30.9037 | 30.0444  | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  30.9037 | 10.3407  | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | SOAP          |  30.9037 |  1.06667 | MILO_LW  |       nan |               | final_validation_accuracy |
| SGD           | ADAMW         |  24.637  | 19.2889  | SGD      |       nan |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  24.637  | 30.0444  | ADAGRAD  |       nan |               | final_validation_accuracy |
| SGD           | ADEMAMIX      |  24.637  | 10.3407  | SGD      |       nan |               | final_validation_accuracy |
| SGD           | SOAP          |  24.637  |  1.06667 | SGD      |       nan |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  19.2889 | 30.0444  | ADAGRAD  |       nan |               | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  19.2889 | 10.3407  | ADAMW    |       nan |               | final_validation_accuracy |
| ADAMW         | SOAP          |  19.2889 |  1.06667 | ADAMW    |       nan |               | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  30.0444 | 10.3407  | ADAGRAD  |       nan |               | final_validation_accuracy |
| ADAGRAD       | SOAP          |  30.0444 |  1.06667 | ADAGRAD  |       nan |               | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  10.3407 |  1.06667 | ADEMAMIX |       nan |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.2969 | 0.0000 | 0.0000 | 0.2969 | 0.2969 |
| MILO_LW | 0.2894 | 0.0000 | 0.0000 | 0.2894 | 0.2894 |
| SGD | 0.2194 | 0.0000 | 0.0000 | 0.2194 | 0.2194 |
| ADAMW | 0.1580 | 0.0000 | 0.0000 | 0.1580 | 0.1580 |
| ADAGRAD | 0.2853 | 0.0000 | 0.0000 | 0.2853 | 0.2853 |
| ADEMAMIX | 0.0660 | 0.0000 | 0.0000 | 0.0660 | 0.0660 |
| SOAP | 0.0002 | 0.0000 | 0.0000 | 0.0002 | 0.0002 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |      Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|----------:|------------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.296926  | 0.289358    | MILO     |       nan |               | final_validation_f1_score |
| MILO          | SGD           | 0.296926  | 0.21935     | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.296926  | 0.158033    | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.296926  | 0.285276    | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.296926  | 0.0659687   | MILO     |       nan |               | final_validation_f1_score |
| MILO          | SOAP          | 0.296926  | 0.000211082 | MILO     |       nan |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.289358  | 0.21935     | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.289358  | 0.158033    | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.289358  | 0.285276    | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.289358  | 0.0659687   | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.289358  | 0.000211082 | MILO_LW  |       nan |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.21935   | 0.158033    | SGD      |       nan |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.21935   | 0.285276    | ADAGRAD  |       nan |               | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.21935   | 0.0659687   | SGD      |       nan |               | final_validation_f1_score |
| SGD           | SOAP          | 0.21935   | 0.000211082 | SGD      |       nan |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.158033  | 0.285276    | ADAGRAD  |       nan |               | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.158033  | 0.0659687   | ADAMW    |       nan |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.158033  | 0.000211082 | ADAMW    |       nan |               | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.285276  | 0.0659687   | ADAGRAD  |       nan |               | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.285276  | 0.000211082 | ADAGRAD  |       nan |               | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.0659687 | 0.000211082 | ADEMAMIX |       nan |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9300 | 0.0000 | 0.0000 | 0.9300 | 0.9300 |
| MILO_LW | 0.9293 | 0.0000 | 0.0000 | 0.9293 | 0.9293 |
| SGD | 0.9124 | 0.0000 | 0.0000 | 0.9124 | 0.9124 |
| ADAMW | 0.9060 | 0.0000 | 0.0000 | 0.9060 | 0.9060 |
| ADAGRAD | 0.9233 | 0.0000 | 0.0000 | 0.9233 | 0.9233 |
| ADEMAMIX | 0.8514 | 0.0000 | 0.0000 | 0.8514 | 0.8514 |
| SOAP | -1.0000 | 0.0000 | 0.0000 | -1.0000 | -1.0000 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |    Mean B | Better   |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|----------:|:---------|----------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.929971 |  0.929337 | MILO     |       nan |               | final_validation_auc |
| MILO          | SGD           | 0.929971 |  0.912429 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADAMW         | 0.929971 |  0.905951 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADAGRAD       | 0.929971 |  0.923342 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADEMAMIX      | 0.929971 |  0.851364 | MILO     |       nan |               | final_validation_auc |
| MILO          | SOAP          | 0.929971 | -1        | MILO     |       nan |               | final_validation_auc |
| MILO_LW       | SGD           | 0.929337 |  0.912429 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.929337 |  0.905951 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.929337 |  0.923342 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.929337 |  0.851364 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | SOAP          | 0.929337 | -1        | MILO_LW  |       nan |               | final_validation_auc |
| SGD           | ADAMW         | 0.912429 |  0.905951 | SGD      |       nan |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.912429 |  0.923342 | ADAGRAD  |       nan |               | final_validation_auc |
| SGD           | ADEMAMIX      | 0.912429 |  0.851364 | SGD      |       nan |               | final_validation_auc |
| SGD           | SOAP          | 0.912429 | -1        | SGD      |       nan |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.905951 |  0.923342 | ADAGRAD  |       nan |               | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.905951 |  0.851364 | ADAMW    |       nan |               | final_validation_auc |
| ADAMW         | SOAP          | 0.905951 | -1        | ADAMW    |       nan |               | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.923342 |  0.851364 | ADAGRAD  |       nan |               | final_validation_auc |
| ADAGRAD       | SOAP          | 0.923342 | -1        | ADAGRAD  |       nan |               | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.851364 | -1        | ADEMAMIX |       nan |               | final_validation_auc |

