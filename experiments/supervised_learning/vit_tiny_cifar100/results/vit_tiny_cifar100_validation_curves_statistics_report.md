# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 3.7098 | 0.0000 | 0.0000 | 3.7098 | 3.7098 |
| MILO_LW | 3.7231 | 0.0000 | 0.0000 | 3.7231 | 3.7231 |
| SGD | 3.6200 | 0.0000 | 0.0000 | 3.6200 | 3.6200 |
| ADAMW | 2.9386 | 0.0000 | 0.0000 | 2.9386 | 2.9386 |
| ADAGRAD | 4.2625 | 0.0000 | 0.0000 | 4.2625 | 4.2625 |
| ADEMAMIX | 2.8954 | 0.0000 | 0.0000 | 2.8954 | 2.8954 |
| SOAP | 2.3103 | 0.0000 | 0.0000 | 2.3103 | 2.3103 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  3.70979 |  3.72309 | MILO     |       nan |               | final_validation_loss |
| MILO          | SGD           |  3.70979 |  3.61998 | SGD      |       nan |               | final_validation_loss |
| MILO          | ADAMW         |  3.70979 |  2.93857 | ADAMW    |       nan |               | final_validation_loss |
| MILO          | ADAGRAD       |  3.70979 |  4.26247 | MILO     |       nan |               | final_validation_loss |
| MILO          | ADEMAMIX      |  3.70979 |  2.89541 | ADEMAMIX |       nan |               | final_validation_loss |
| MILO          | SOAP          |  3.70979 |  2.31027 | SOAP     |       nan |               | final_validation_loss |
| MILO_LW       | SGD           |  3.72309 |  3.61998 | SGD      |       nan |               | final_validation_loss |
| MILO_LW       | ADAMW         |  3.72309 |  2.93857 | ADAMW    |       nan |               | final_validation_loss |
| MILO_LW       | ADAGRAD       |  3.72309 |  4.26247 | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADEMAMIX      |  3.72309 |  2.89541 | ADEMAMIX |       nan |               | final_validation_loss |
| MILO_LW       | SOAP          |  3.72309 |  2.31027 | SOAP     |       nan |               | final_validation_loss |
| SGD           | ADAMW         |  3.61998 |  2.93857 | ADAMW    |       nan |               | final_validation_loss |
| SGD           | ADAGRAD       |  3.61998 |  4.26247 | SGD      |       nan |               | final_validation_loss |
| SGD           | ADEMAMIX      |  3.61998 |  2.89541 | ADEMAMIX |       nan |               | final_validation_loss |
| SGD           | SOAP          |  3.61998 |  2.31027 | SOAP     |       nan |               | final_validation_loss |
| ADAMW         | ADAGRAD       |  2.93857 |  4.26247 | ADAMW    |       nan |               | final_validation_loss |
| ADAMW         | ADEMAMIX      |  2.93857 |  2.89541 | ADEMAMIX |       nan |               | final_validation_loss |
| ADAMW         | SOAP          |  2.93857 |  2.31027 | SOAP     |       nan |               | final_validation_loss |
| ADAGRAD       | ADEMAMIX      |  4.26247 |  2.89541 | ADEMAMIX |       nan |               | final_validation_loss |
| ADAGRAD       | SOAP          |  4.26247 |  2.31027 | SOAP     |       nan |               | final_validation_loss |
| ADEMAMIX      | SOAP          |  2.89541 |  2.31027 | SOAP     |       nan |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 16.3704 | 0.0000 | 0.0000 | 16.3704 | 16.3704 |
| MILO_LW | 16.0741 | 0.0000 | 0.0000 | 16.0741 | 16.0741 |
| SGD | 15.8074 | 0.0000 | 0.0000 | 15.8074 | 15.8074 |
| ADAMW | 25.9704 | 0.0000 | 0.0000 | 25.9704 | 25.9704 |
| ADAGRAD | 6.6370 | 0.0000 | 0.0000 | 6.6370 | 6.6370 |
| ADEMAMIX | 26.4148 | 0.0000 | 0.0000 | 26.4148 | 26.4148 |
| SOAP | 40.2074 | 0.0000 | 0.0000 | 40.2074 | 40.2074 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 16.3704  | 16.0741  | MILO     |       nan |               | final_validation_accuracy |
| MILO          | SGD           | 16.3704  | 15.8074  | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAMW         | 16.3704  | 25.9704  | ADAMW    |       nan |               | final_validation_accuracy |
| MILO          | ADAGRAD       | 16.3704  |  6.63704 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADEMAMIX      | 16.3704  | 26.4148  | ADEMAMIX |       nan |               | final_validation_accuracy |
| MILO          | SOAP          | 16.3704  | 40.2074  | SOAP     |       nan |               | final_validation_accuracy |
| MILO_LW       | SGD           | 16.0741  | 15.8074  | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAMW         | 16.0741  | 25.9704  | ADAMW    |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAGRAD       | 16.0741  |  6.63704 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      | 16.0741  | 26.4148  | ADEMAMIX |       nan |               | final_validation_accuracy |
| MILO_LW       | SOAP          | 16.0741  | 40.2074  | SOAP     |       nan |               | final_validation_accuracy |
| SGD           | ADAMW         | 15.8074  | 25.9704  | ADAMW    |       nan |               | final_validation_accuracy |
| SGD           | ADAGRAD       | 15.8074  |  6.63704 | SGD      |       nan |               | final_validation_accuracy |
| SGD           | ADEMAMIX      | 15.8074  | 26.4148  | ADEMAMIX |       nan |               | final_validation_accuracy |
| SGD           | SOAP          | 15.8074  | 40.2074  | SOAP     |       nan |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       | 25.9704  |  6.63704 | ADAMW    |       nan |               | final_validation_accuracy |
| ADAMW         | ADEMAMIX      | 25.9704  | 26.4148  | ADEMAMIX |       nan |               | final_validation_accuracy |
| ADAMW         | SOAP          | 25.9704  | 40.2074  | SOAP     |       nan |               | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  6.63704 | 26.4148  | ADEMAMIX |       nan |               | final_validation_accuracy |
| ADAGRAD       | SOAP          |  6.63704 | 40.2074  | SOAP     |       nan |               | final_validation_accuracy |
| ADEMAMIX      | SOAP          | 26.4148  | 40.2074  | SOAP     |       nan |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.1240 | 0.0000 | 0.0000 | 0.1240 | 0.1240 |
| MILO_LW | 0.1253 | 0.0000 | 0.0000 | 0.1253 | 0.1253 |
| SGD | 0.1245 | 0.0000 | 0.0000 | 0.1245 | 0.1245 |
| ADAMW | 0.2453 | 0.0000 | 0.0000 | 0.2453 | 0.2453 |
| ADAGRAD | 0.0360 | 0.0000 | 0.0000 | 0.0360 | 0.0360 |
| ADEMAMIX | 0.2494 | 0.0000 | 0.0000 | 0.2494 | 0.2494 |
| SOAP | 0.3959 | 0.0000 | 0.0000 | 0.3959 | 0.3959 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |    Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|----------:|----------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.124049  | 0.1253    | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO          | SGD           | 0.124049  | 0.124528  | SGD      |       nan |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.124049  | 0.245349  | ADAMW    |       nan |               | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.124049  | 0.0360247 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.124049  | 0.249425  | ADEMAMIX |       nan |               | final_validation_f1_score |
| MILO          | SOAP          | 0.124049  | 0.395891  | SOAP     |       nan |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.1253    | 0.124528  | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.1253    | 0.245349  | ADAMW    |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.1253    | 0.0360247 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.1253    | 0.249425  | ADEMAMIX |       nan |               | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.1253    | 0.395891  | SOAP     |       nan |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.124528  | 0.245349  | ADAMW    |       nan |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.124528  | 0.0360247 | SGD      |       nan |               | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.124528  | 0.249425  | ADEMAMIX |       nan |               | final_validation_f1_score |
| SGD           | SOAP          | 0.124528  | 0.395891  | SOAP     |       nan |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.245349  | 0.0360247 | ADAMW    |       nan |               | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.245349  | 0.249425  | ADEMAMIX |       nan |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.245349  | 0.395891  | SOAP     |       nan |               | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.0360247 | 0.249425  | ADEMAMIX |       nan |               | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.0360247 | 0.395891  | SOAP     |       nan |               | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.249425  | 0.395891  | SOAP     |       nan |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.8559 | 0.0000 | 0.0000 | 0.8559 | 0.8559 |
| MILO_LW | 0.8548 | 0.0000 | 0.0000 | 0.8548 | 0.8548 |
| SGD | 0.8455 | 0.0000 | 0.0000 | 0.8455 | 0.8455 |
| ADAMW | 0.9128 | 0.0000 | 0.0000 | 0.9128 | 0.9128 |
| ADAGRAD | 0.7541 | 0.0000 | 0.0000 | 0.7541 | 0.7541 |
| ADEMAMIX | 0.9186 | 0.0000 | 0.0000 | 0.9186 | 0.9186 |
| SOAP | 0.9522 | 0.0000 | 0.0000 | 0.9522 | 0.9522 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.855942 | 0.854833 | MILO     |       nan |               | final_validation_auc |
| MILO          | SGD           | 0.855942 | 0.845466 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADAMW         | 0.855942 | 0.912832 | ADAMW    |       nan |               | final_validation_auc |
| MILO          | ADAGRAD       | 0.855942 | 0.754061 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADEMAMIX      | 0.855942 | 0.918551 | ADEMAMIX |       nan |               | final_validation_auc |
| MILO          | SOAP          | 0.855942 | 0.952201 | SOAP     |       nan |               | final_validation_auc |
| MILO_LW       | SGD           | 0.854833 | 0.845466 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.854833 | 0.912832 | ADAMW    |       nan |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.854833 | 0.754061 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.854833 | 0.918551 | ADEMAMIX |       nan |               | final_validation_auc |
| MILO_LW       | SOAP          | 0.854833 | 0.952201 | SOAP     |       nan |               | final_validation_auc |
| SGD           | ADAMW         | 0.845466 | 0.912832 | ADAMW    |       nan |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.845466 | 0.754061 | SGD      |       nan |               | final_validation_auc |
| SGD           | ADEMAMIX      | 0.845466 | 0.918551 | ADEMAMIX |       nan |               | final_validation_auc |
| SGD           | SOAP          | 0.845466 | 0.952201 | SOAP     |       nan |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.912832 | 0.754061 | ADAMW    |       nan |               | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.912832 | 0.918551 | ADEMAMIX |       nan |               | final_validation_auc |
| ADAMW         | SOAP          | 0.912832 | 0.952201 | SOAP     |       nan |               | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.754061 | 0.918551 | ADEMAMIX |       nan |               | final_validation_auc |
| ADAGRAD       | SOAP          | 0.754061 | 0.952201 | SOAP     |       nan |               | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.918551 | 0.952201 | SOAP     |       nan |               | final_validation_auc |

