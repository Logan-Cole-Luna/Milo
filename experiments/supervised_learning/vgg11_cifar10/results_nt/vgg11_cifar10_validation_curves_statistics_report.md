# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.8833 | 0.0000 | 0.0000 | 0.8833 | 0.8833 |
| MILO_LW | 0.8480 | 0.0000 | 0.0000 | 0.8480 | 0.8480 |
| SGD | 0.9659 | 0.0000 | 0.0000 | 0.9659 | 0.9659 |
| ADAMW | 1.0461 | 0.0000 | 0.0000 | 1.0461 | 1.0461 |
| ADAGRAD | 0.9418 | 0.0000 | 0.0000 | 0.9418 | 0.9418 |
| ADEMAMIX | 0.9164 | 0.0000 | 0.0000 | 0.9164 | 0.9164 |
| SOAP | 1400133.3575 | 0.0000 | 0.0000 | 1400133.3575 | 1400133.3575 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |      Mean B | Better   |   p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|------------:|:---------|----------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.883317 | 0.847979    | MILO_LW  |       nan |               | final_validation_loss |
| MILO          | SGD           | 0.883317 | 0.965871    | MILO     |       nan |               | final_validation_loss |
| MILO          | ADAMW         | 0.883317 | 1.04609     | MILO     |       nan |               | final_validation_loss |
| MILO          | ADAGRAD       | 0.883317 | 0.941799    | MILO     |       nan |               | final_validation_loss |
| MILO          | ADEMAMIX      | 0.883317 | 0.916443    | MILO     |       nan |               | final_validation_loss |
| MILO          | SOAP          | 0.883317 | 1.40013e+06 | MILO     |       nan |               | final_validation_loss |
| MILO_LW       | SGD           | 0.847979 | 0.965871    | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADAMW         | 0.847979 | 1.04609     | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.847979 | 0.941799    | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.847979 | 0.916443    | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | SOAP          | 0.847979 | 1.40013e+06 | MILO_LW  |       nan |               | final_validation_loss |
| SGD           | ADAMW         | 0.965871 | 1.04609     | SGD      |       nan |               | final_validation_loss |
| SGD           | ADAGRAD       | 0.965871 | 0.941799    | ADAGRAD  |       nan |               | final_validation_loss |
| SGD           | ADEMAMIX      | 0.965871 | 0.916443    | ADEMAMIX |       nan |               | final_validation_loss |
| SGD           | SOAP          | 0.965871 | 1.40013e+06 | SGD      |       nan |               | final_validation_loss |
| ADAMW         | ADAGRAD       | 1.04609  | 0.941799    | ADAGRAD  |       nan |               | final_validation_loss |
| ADAMW         | ADEMAMIX      | 1.04609  | 0.916443    | ADEMAMIX |       nan |               | final_validation_loss |
| ADAMW         | SOAP          | 1.04609  | 1.40013e+06 | ADAMW    |       nan |               | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 0.941799 | 0.916443    | ADEMAMIX |       nan |               | final_validation_loss |
| ADAGRAD       | SOAP          | 0.941799 | 1.40013e+06 | ADAGRAD  |       nan |               | final_validation_loss |
| ADEMAMIX      | SOAP          | 0.916443 | 1.40013e+06 | ADEMAMIX |       nan |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 71.4370 | 0.0000 | 0.0000 | 71.4370 | 71.4370 |
| MILO_LW | 73.2889 | 0.0000 | 0.0000 | 73.2889 | 73.2889 |
| SGD | 70.2222 | 0.0000 | 0.0000 | 70.2222 | 70.2222 |
| ADAMW | 66.1037 | 0.0000 | 0.0000 | 66.1037 | 66.1037 |
| ADAGRAD | 72.2222 | 0.0000 | 0.0000 | 72.2222 | 72.2222 |
| ADEMAMIX | 68.3407 | 0.0000 | 0.0000 | 68.3407 | 68.3407 |
| SOAP | 10.1778 | 0.0000 | 0.0000 | 10.1778 | 10.1778 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  71.437  |  73.2889 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO          | SGD           |  71.437  |  70.2222 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAMW         |  71.437  |  66.1037 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAGRAD       |  71.437  |  72.2222 | ADAGRAD  |       nan |               | final_validation_accuracy |
| MILO          | ADEMAMIX      |  71.437  |  68.3407 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | SOAP          |  71.437  |  10.1778 | MILO     |       nan |               | final_validation_accuracy |
| MILO_LW       | SGD           |  73.2889 |  70.2222 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  73.2889 |  66.1037 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  73.2889 |  72.2222 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  73.2889 |  68.3407 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | SOAP          |  73.2889 |  10.1778 | MILO_LW  |       nan |               | final_validation_accuracy |
| SGD           | ADAMW         |  70.2222 |  66.1037 | SGD      |       nan |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  70.2222 |  72.2222 | ADAGRAD  |       nan |               | final_validation_accuracy |
| SGD           | ADEMAMIX      |  70.2222 |  68.3407 | SGD      |       nan |               | final_validation_accuracy |
| SGD           | SOAP          |  70.2222 |  10.1778 | SGD      |       nan |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  66.1037 |  72.2222 | ADAGRAD  |       nan |               | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  66.1037 |  68.3407 | ADEMAMIX |       nan |               | final_validation_accuracy |
| ADAMW         | SOAP          |  66.1037 |  10.1778 | ADAMW    |       nan |               | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  72.2222 |  68.3407 | ADAGRAD  |       nan |               | final_validation_accuracy |
| ADAGRAD       | SOAP          |  72.2222 |  10.1778 | ADAGRAD  |       nan |               | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  68.3407 |  10.1778 | ADEMAMIX |       nan |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.7142 | 0.0000 | 0.0000 | 0.7142 | 0.7142 |
| MILO_LW | 0.7308 | 0.0000 | 0.0000 | 0.7308 | 0.7308 |
| SGD | 0.7038 | 0.0000 | 0.0000 | 0.7038 | 0.7038 |
| ADAMW | 0.6700 | 0.0000 | 0.0000 | 0.6700 | 0.6700 |
| ADAGRAD | 0.7251 | 0.0000 | 0.0000 | 0.7251 | 0.7251 |
| ADEMAMIX | 0.6817 | 0.0000 | 0.0000 | 0.6817 | 0.6817 |
| SOAP | 0.0221 | 0.0000 | 0.0000 | 0.0221 | 0.0221 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |    Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|----------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.714229 | 0.730792  | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO          | SGD           | 0.714229 | 0.703829  | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.714229 | 0.670002  | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.714229 | 0.725125  | ADAGRAD  |       nan |               | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.714229 | 0.681701  | MILO     |       nan |               | final_validation_f1_score |
| MILO          | SOAP          | 0.714229 | 0.0221412 | MILO     |       nan |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.730792 | 0.703829  | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.730792 | 0.670002  | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.730792 | 0.725125  | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.730792 | 0.681701  | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.730792 | 0.0221412 | MILO_LW  |       nan |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.703829 | 0.670002  | SGD      |       nan |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.703829 | 0.725125  | ADAGRAD  |       nan |               | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.703829 | 0.681701  | SGD      |       nan |               | final_validation_f1_score |
| SGD           | SOAP          | 0.703829 | 0.0221412 | SGD      |       nan |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.670002 | 0.725125  | ADAGRAD  |       nan |               | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.670002 | 0.681701  | ADEMAMIX |       nan |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.670002 | 0.0221412 | ADAMW    |       nan |               | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.725125 | 0.681701  | ADAGRAD  |       nan |               | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.725125 | 0.0221412 | ADAGRAD  |       nan |               | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.681701 | 0.0221412 | ADEMAMIX |       nan |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9603 | 0.0000 | 0.0000 | 0.9603 | 0.9603 |
| MILO_LW | 0.9625 | 0.0000 | 0.0000 | 0.9625 | 0.9625 |
| SGD | 0.9590 | 0.0000 | 0.0000 | 0.9590 | 0.9590 |
| ADAMW | 0.9424 | 0.0000 | 0.0000 | 0.9424 | 0.9424 |
| ADAGRAD | 0.9617 | 0.0000 | 0.0000 | 0.9617 | 0.9617 |
| ADEMAMIX | 0.9522 | 0.0000 | 0.0000 | 0.9522 | 0.9522 |
| SOAP | 0.5024 | 0.0000 | 0.0000 | 0.5024 | 0.5024 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.960341 | 0.962518 | MILO_LW  |       nan |               | final_validation_auc |
| MILO          | SGD           | 0.960341 | 0.958975 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADAMW         | 0.960341 | 0.942352 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADAGRAD       | 0.960341 | 0.961726 | ADAGRAD  |       nan |               | final_validation_auc |
| MILO          | ADEMAMIX      | 0.960341 | 0.95217  | MILO     |       nan |               | final_validation_auc |
| MILO          | SOAP          | 0.960341 | 0.502353 | MILO     |       nan |               | final_validation_auc |
| MILO_LW       | SGD           | 0.962518 | 0.958975 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.962518 | 0.942352 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.962518 | 0.961726 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.962518 | 0.95217  | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | SOAP          | 0.962518 | 0.502353 | MILO_LW  |       nan |               | final_validation_auc |
| SGD           | ADAMW         | 0.958975 | 0.942352 | SGD      |       nan |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.958975 | 0.961726 | ADAGRAD  |       nan |               | final_validation_auc |
| SGD           | ADEMAMIX      | 0.958975 | 0.95217  | SGD      |       nan |               | final_validation_auc |
| SGD           | SOAP          | 0.958975 | 0.502353 | SGD      |       nan |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.942352 | 0.961726 | ADAGRAD  |       nan |               | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.942352 | 0.95217  | ADEMAMIX |       nan |               | final_validation_auc |
| ADAMW         | SOAP          | 0.942352 | 0.502353 | ADAMW    |       nan |               | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.961726 | 0.95217  | ADAGRAD  |       nan |               | final_validation_auc |
| ADAGRAD       | SOAP          | 0.961726 | 0.502353 | ADAGRAD  |       nan |               | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.95217  | 0.502353 | ADEMAMIX |       nan |               | final_validation_auc |

