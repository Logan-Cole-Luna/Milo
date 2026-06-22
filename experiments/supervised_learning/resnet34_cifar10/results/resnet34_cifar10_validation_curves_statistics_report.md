# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.7583 | 0.0000 | 0.0000 | 0.7583 | 0.7583 |
| MILO_LW | 0.7699 | 0.0000 | 0.0000 | 0.7699 | 0.7699 |
| SGD | 0.9963 | 0.0000 | 0.0000 | 0.9963 | 0.9963 |
| ADAMW | 0.7184 | 0.0000 | 0.0000 | 0.7184 | 0.7184 |
| ADAGRAD | 1.0535 | 0.0000 | 0.0000 | 1.0535 | 1.0535 |
| ADEMAMIX | 0.7062 | 0.0000 | 0.0000 | 0.7062 | 0.7062 |
| SOAP | 0.5078 | 0.0000 | 0.0000 | 0.5078 | 0.5078 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.758325 | 0.769885 | MILO     |       nan |               | final_validation_loss |
| MILO          | SGD           | 0.758325 | 0.996339 | MILO     |       nan |               | final_validation_loss |
| MILO          | ADAMW         | 0.758325 | 0.718398 | ADAMW    |       nan |               | final_validation_loss |
| MILO          | ADAGRAD       | 0.758325 | 1.0535   | MILO     |       nan |               | final_validation_loss |
| MILO          | ADEMAMIX      | 0.758325 | 0.706156 | ADEMAMIX |       nan |               | final_validation_loss |
| MILO          | SOAP          | 0.758325 | 0.507819 | SOAP     |       nan |               | final_validation_loss |
| MILO_LW       | SGD           | 0.769885 | 0.996339 | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADAMW         | 0.769885 | 0.718398 | ADAMW    |       nan |               | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.769885 | 1.0535   | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.769885 | 0.706156 | ADEMAMIX |       nan |               | final_validation_loss |
| MILO_LW       | SOAP          | 0.769885 | 0.507819 | SOAP     |       nan |               | final_validation_loss |
| SGD           | ADAMW         | 0.996339 | 0.718398 | ADAMW    |       nan |               | final_validation_loss |
| SGD           | ADAGRAD       | 0.996339 | 1.0535   | SGD      |       nan |               | final_validation_loss |
| SGD           | ADEMAMIX      | 0.996339 | 0.706156 | ADEMAMIX |       nan |               | final_validation_loss |
| SGD           | SOAP          | 0.996339 | 0.507819 | SOAP     |       nan |               | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.718398 | 1.0535   | ADAMW    |       nan |               | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.718398 | 0.706156 | ADEMAMIX |       nan |               | final_validation_loss |
| ADAMW         | SOAP          | 0.718398 | 0.507819 | SOAP     |       nan |               | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 1.0535   | 0.706156 | ADEMAMIX |       nan |               | final_validation_loss |
| ADAGRAD       | SOAP          | 1.0535   | 0.507819 | SOAP     |       nan |               | final_validation_loss |
| ADEMAMIX      | SOAP          | 0.706156 | 0.507819 | SOAP     |       nan |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 75.0667 | 0.0000 | 0.0000 | 75.0667 | 75.0667 |
| MILO_LW | 74.3407 | 0.0000 | 0.0000 | 74.3407 | 74.3407 |
| SGD | 67.0222 | 0.0000 | 0.0000 | 67.0222 | 67.0222 |
| ADAMW | 76.3111 | 0.0000 | 0.0000 | 76.3111 | 76.3111 |
| ADAGRAD | 62.9037 | 0.0000 | 0.0000 | 62.9037 | 62.9037 |
| ADEMAMIX | 75.4667 | 0.0000 | 0.0000 | 75.4667 | 75.4667 |
| SOAP | 85.8370 | 0.0000 | 0.0000 | 85.8370 | 85.8370 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  75.0667 |  74.3407 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | SGD           |  75.0667 |  67.0222 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAMW         |  75.0667 |  76.3111 | ADAMW    |       nan |               | final_validation_accuracy |
| MILO          | ADAGRAD       |  75.0667 |  62.9037 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADEMAMIX      |  75.0667 |  75.4667 | ADEMAMIX |       nan |               | final_validation_accuracy |
| MILO          | SOAP          |  75.0667 |  85.837  | SOAP     |       nan |               | final_validation_accuracy |
| MILO_LW       | SGD           |  74.3407 |  67.0222 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  74.3407 |  76.3111 | ADAMW    |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  74.3407 |  62.9037 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  74.3407 |  75.4667 | ADEMAMIX |       nan |               | final_validation_accuracy |
| MILO_LW       | SOAP          |  74.3407 |  85.837  | SOAP     |       nan |               | final_validation_accuracy |
| SGD           | ADAMW         |  67.0222 |  76.3111 | ADAMW    |       nan |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  67.0222 |  62.9037 | SGD      |       nan |               | final_validation_accuracy |
| SGD           | ADEMAMIX      |  67.0222 |  75.4667 | ADEMAMIX |       nan |               | final_validation_accuracy |
| SGD           | SOAP          |  67.0222 |  85.837  | SOAP     |       nan |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  76.3111 |  62.9037 | ADAMW    |       nan |               | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  76.3111 |  75.4667 | ADAMW    |       nan |               | final_validation_accuracy |
| ADAMW         | SOAP          |  76.3111 |  85.837  | SOAP     |       nan |               | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  62.9037 |  75.4667 | ADEMAMIX |       nan |               | final_validation_accuracy |
| ADAGRAD       | SOAP          |  62.9037 |  85.837  | SOAP     |       nan |               | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  75.4667 |  85.837  | SOAP     |       nan |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.7495 | 0.0000 | 0.0000 | 0.7495 | 0.7495 |
| MILO_LW | 0.7421 | 0.0000 | 0.0000 | 0.7421 | 0.7421 |
| SGD | 0.6608 | 0.0000 | 0.0000 | 0.6608 | 0.6608 |
| ADAMW | 0.7598 | 0.0000 | 0.0000 | 0.7598 | 0.7598 |
| ADAGRAD | 0.6260 | 0.0000 | 0.0000 | 0.6260 | 0.6260 |
| ADEMAMIX | 0.7561 | 0.0000 | 0.0000 | 0.7561 | 0.7561 |
| SOAP | 0.8587 | 0.0000 | 0.0000 | 0.8587 | 0.8587 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.749523 | 0.742073 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | SGD           | 0.749523 | 0.660803 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.749523 | 0.759827 | ADAMW    |       nan |               | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.749523 | 0.626027 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.749523 | 0.756064 | ADEMAMIX |       nan |               | final_validation_f1_score |
| MILO          | SOAP          | 0.749523 | 0.85869  | SOAP     |       nan |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.742073 | 0.660803 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.742073 | 0.759827 | ADAMW    |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.742073 | 0.626027 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.742073 | 0.756064 | ADEMAMIX |       nan |               | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.742073 | 0.85869  | SOAP     |       nan |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.660803 | 0.759827 | ADAMW    |       nan |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.660803 | 0.626027 | SGD      |       nan |               | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.660803 | 0.756064 | ADEMAMIX |       nan |               | final_validation_f1_score |
| SGD           | SOAP          | 0.660803 | 0.85869  | SOAP     |       nan |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.759827 | 0.626027 | ADAMW    |       nan |               | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.759827 | 0.756064 | ADAMW    |       nan |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.759827 | 0.85869  | SOAP     |       nan |               | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.626027 | 0.756064 | ADEMAMIX |       nan |               | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.626027 | 0.85869  | SOAP     |       nan |               | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.756064 | 0.85869  | SOAP     |       nan |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9678 | 0.0000 | 0.0000 | 0.9678 | 0.9678 |
| MILO_LW | 0.9665 | 0.0000 | 0.0000 | 0.9665 | 0.9665 |
| SGD | 0.9549 | 0.0000 | 0.0000 | 0.9549 | 0.9549 |
| ADAMW | 0.9776 | 0.0000 | 0.0000 | 0.9776 | 0.9776 |
| ADAGRAD | 0.9336 | 0.0000 | 0.0000 | 0.9336 | 0.9336 |
| ADEMAMIX | 0.9717 | 0.0000 | 0.0000 | 0.9717 | 0.9717 |
| SOAP | 0.9869 | 0.0000 | 0.0000 | 0.9869 | 0.9869 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.967789 | 0.966468 | MILO     |       nan |               | final_validation_auc |
| MILO          | SGD           | 0.967789 | 0.954889 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADAMW         | 0.967789 | 0.977564 | ADAMW    |       nan |               | final_validation_auc |
| MILO          | ADAGRAD       | 0.967789 | 0.93363  | MILO     |       nan |               | final_validation_auc |
| MILO          | ADEMAMIX      | 0.967789 | 0.971717 | ADEMAMIX |       nan |               | final_validation_auc |
| MILO          | SOAP          | 0.967789 | 0.986891 | SOAP     |       nan |               | final_validation_auc |
| MILO_LW       | SGD           | 0.966468 | 0.954889 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.966468 | 0.977564 | ADAMW    |       nan |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.966468 | 0.93363  | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.966468 | 0.971717 | ADEMAMIX |       nan |               | final_validation_auc |
| MILO_LW       | SOAP          | 0.966468 | 0.986891 | SOAP     |       nan |               | final_validation_auc |
| SGD           | ADAMW         | 0.954889 | 0.977564 | ADAMW    |       nan |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.954889 | 0.93363  | SGD      |       nan |               | final_validation_auc |
| SGD           | ADEMAMIX      | 0.954889 | 0.971717 | ADEMAMIX |       nan |               | final_validation_auc |
| SGD           | SOAP          | 0.954889 | 0.986891 | SOAP     |       nan |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.977564 | 0.93363  | ADAMW    |       nan |               | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.977564 | 0.971717 | ADAMW    |       nan |               | final_validation_auc |
| ADAMW         | SOAP          | 0.977564 | 0.986891 | SOAP     |       nan |               | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.93363  | 0.971717 | ADEMAMIX |       nan |               | final_validation_auc |
| ADAGRAD       | SOAP          | 0.93363  | 0.986891 | SOAP     |       nan |               | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.971717 | 0.986891 | SOAP     |       nan |               | final_validation_auc |

