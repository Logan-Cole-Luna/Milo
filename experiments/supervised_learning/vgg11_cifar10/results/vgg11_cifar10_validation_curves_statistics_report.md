# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.8679 | 0.0000 | 0.0000 | 0.8679 | 0.8679 |
| MILO_LW | 0.8723 | 0.0000 | 0.0000 | 0.8723 | 0.8723 |
| SGD | 1.0839 | 0.0000 | 0.0000 | 1.0839 | 1.0839 |
| ADAMW | 0.9958 | 0.0000 | 0.0000 | 0.9958 | 0.9958 |
| ADAGRAD | 1.2450 | 0.0000 | 0.0000 | 1.2450 | 1.2450 |
| ADEMAMIX | 0.9663 | 0.0000 | 0.0000 | 0.9663 | 0.9663 |
| SOAP | 0.8404 | 0.0000 | 0.0000 | 0.8404 | 0.8404 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.867894 | 0.872311 | MILO     |       nan |               | final_validation_loss |
| MILO          | SGD           | 0.867894 | 1.08386  | MILO     |       nan |               | final_validation_loss |
| MILO          | ADAMW         | 0.867894 | 0.995846 | MILO     |       nan |               | final_validation_loss |
| MILO          | ADAGRAD       | 0.867894 | 1.24501  | MILO     |       nan |               | final_validation_loss |
| MILO          | ADEMAMIX      | 0.867894 | 0.966314 | MILO     |       nan |               | final_validation_loss |
| MILO          | SOAP          | 0.867894 | 0.840422 | SOAP     |       nan |               | final_validation_loss |
| MILO_LW       | SGD           | 0.872311 | 1.08386  | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADAMW         | 0.872311 | 0.995846 | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.872311 | 1.24501  | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.872311 | 0.966314 | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | SOAP          | 0.872311 | 0.840422 | SOAP     |       nan |               | final_validation_loss |
| SGD           | ADAMW         | 1.08386  | 0.995846 | ADAMW    |       nan |               | final_validation_loss |
| SGD           | ADAGRAD       | 1.08386  | 1.24501  | SGD      |       nan |               | final_validation_loss |
| SGD           | ADEMAMIX      | 1.08386  | 0.966314 | ADEMAMIX |       nan |               | final_validation_loss |
| SGD           | SOAP          | 1.08386  | 0.840422 | SOAP     |       nan |               | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.995846 | 1.24501  | ADAMW    |       nan |               | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.995846 | 0.966314 | ADEMAMIX |       nan |               | final_validation_loss |
| ADAMW         | SOAP          | 0.995846 | 0.840422 | SOAP     |       nan |               | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 1.24501  | 0.966314 | ADEMAMIX |       nan |               | final_validation_loss |
| ADAGRAD       | SOAP          | 1.24501  | 0.840422 | SOAP     |       nan |               | final_validation_loss |
| ADEMAMIX      | SOAP          | 0.966314 | 0.840422 | SOAP     |       nan |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 71.4963 | 0.0000 | 0.0000 | 71.4963 | 71.4963 |
| MILO_LW | 72.6074 | 0.0000 | 0.0000 | 72.6074 | 72.6074 |
| SGD | 61.9407 | 0.0000 | 0.0000 | 61.9407 | 61.9407 |
| ADAMW | 65.6444 | 0.0000 | 0.0000 | 65.6444 | 65.6444 |
| ADAGRAD | 54.1037 | 0.0000 | 0.0000 | 54.1037 | 54.1037 |
| ADEMAMIX | 66.7407 | 0.0000 | 0.0000 | 66.7407 | 66.7407 |
| SOAP | 79.2296 | 0.0000 | 0.0000 | 79.2296 | 79.2296 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  71.4963 |  72.6074 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO          | SGD           |  71.4963 |  61.9407 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAMW         |  71.4963 |  65.6444 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAGRAD       |  71.4963 |  54.1037 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADEMAMIX      |  71.4963 |  66.7407 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | SOAP          |  71.4963 |  79.2296 | SOAP     |       nan |               | final_validation_accuracy |
| MILO_LW       | SGD           |  72.6074 |  61.9407 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  72.6074 |  65.6444 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  72.6074 |  54.1037 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  72.6074 |  66.7407 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | SOAP          |  72.6074 |  79.2296 | SOAP     |       nan |               | final_validation_accuracy |
| SGD           | ADAMW         |  61.9407 |  65.6444 | ADAMW    |       nan |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  61.9407 |  54.1037 | SGD      |       nan |               | final_validation_accuracy |
| SGD           | ADEMAMIX      |  61.9407 |  66.7407 | ADEMAMIX |       nan |               | final_validation_accuracy |
| SGD           | SOAP          |  61.9407 |  79.2296 | SOAP     |       nan |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  65.6444 |  54.1037 | ADAMW    |       nan |               | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  65.6444 |  66.7407 | ADEMAMIX |       nan |               | final_validation_accuracy |
| ADAMW         | SOAP          |  65.6444 |  79.2296 | SOAP     |       nan |               | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  54.1037 |  66.7407 | ADEMAMIX |       nan |               | final_validation_accuracy |
| ADAGRAD       | SOAP          |  54.1037 |  79.2296 | SOAP     |       nan |               | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  66.7407 |  79.2296 | SOAP     |       nan |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.7151 | 0.0000 | 0.0000 | 0.7151 | 0.7151 |
| MILO_LW | 0.7248 | 0.0000 | 0.0000 | 0.7248 | 0.7248 |
| SGD | 0.6102 | 0.0000 | 0.0000 | 0.6102 | 0.6102 |
| ADAMW | 0.6394 | 0.0000 | 0.0000 | 0.6394 | 0.6394 |
| ADAGRAD | 0.5335 | 0.0000 | 0.0000 | 0.5335 | 0.5335 |
| ADEMAMIX | 0.6672 | 0.0000 | 0.0000 | 0.6672 | 0.6672 |
| SOAP | 0.7935 | 0.0000 | 0.0000 | 0.7935 | 0.7935 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.715085 | 0.724807 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO          | SGD           | 0.715085 | 0.610161 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.715085 | 0.639394 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.715085 | 0.533461 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.715085 | 0.667195 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | SOAP          | 0.715085 | 0.793462 | SOAP     |       nan |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.724807 | 0.610161 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.724807 | 0.639394 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.724807 | 0.533461 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.724807 | 0.667195 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.724807 | 0.793462 | SOAP     |       nan |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.610161 | 0.639394 | ADAMW    |       nan |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.610161 | 0.533461 | SGD      |       nan |               | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.610161 | 0.667195 | ADEMAMIX |       nan |               | final_validation_f1_score |
| SGD           | SOAP          | 0.610161 | 0.793462 | SOAP     |       nan |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.639394 | 0.533461 | ADAMW    |       nan |               | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.639394 | 0.667195 | ADEMAMIX |       nan |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.639394 | 0.793462 | SOAP     |       nan |               | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.533461 | 0.667195 | ADEMAMIX |       nan |               | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.533461 | 0.793462 | SOAP     |       nan |               | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.667195 | 0.793462 | SOAP     |       nan |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9596 | 0.0000 | 0.0000 | 0.9596 | 0.9596 |
| MILO_LW | 0.9619 | 0.0000 | 0.0000 | 0.9619 | 0.9619 |
| SGD | 0.9375 | 0.0000 | 0.0000 | 0.9375 | 0.9375 |
| ADAMW | 0.9438 | 0.0000 | 0.0000 | 0.9438 | 0.9438 |
| ADAGRAD | 0.9075 | 0.0000 | 0.0000 | 0.9075 | 0.9075 |
| ADEMAMIX | 0.9482 | 0.0000 | 0.0000 | 0.9482 | 0.9482 |
| SOAP | 0.9708 | 0.0000 | 0.0000 | 0.9708 | 0.9708 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.959573 | 0.961918 | MILO_LW  |       nan |               | final_validation_auc |
| MILO          | SGD           | 0.959573 | 0.937522 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADAMW         | 0.959573 | 0.94375  | MILO     |       nan |               | final_validation_auc |
| MILO          | ADAGRAD       | 0.959573 | 0.907508 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADEMAMIX      | 0.959573 | 0.948179 | MILO     |       nan |               | final_validation_auc |
| MILO          | SOAP          | 0.959573 | 0.970768 | SOAP     |       nan |               | final_validation_auc |
| MILO_LW       | SGD           | 0.961918 | 0.937522 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.961918 | 0.94375  | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.961918 | 0.907508 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.961918 | 0.948179 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | SOAP          | 0.961918 | 0.970768 | SOAP     |       nan |               | final_validation_auc |
| SGD           | ADAMW         | 0.937522 | 0.94375  | ADAMW    |       nan |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.937522 | 0.907508 | SGD      |       nan |               | final_validation_auc |
| SGD           | ADEMAMIX      | 0.937522 | 0.948179 | ADEMAMIX |       nan |               | final_validation_auc |
| SGD           | SOAP          | 0.937522 | 0.970768 | SOAP     |       nan |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.94375  | 0.907508 | ADAMW    |       nan |               | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.94375  | 0.948179 | ADEMAMIX |       nan |               | final_validation_auc |
| ADAMW         | SOAP          | 0.94375  | 0.970768 | SOAP     |       nan |               | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.907508 | 0.948179 | ADEMAMIX |       nan |               | final_validation_auc |
| ADAGRAD       | SOAP          | 0.907508 | 0.970768 | SOAP     |       nan |               | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.948179 | 0.970768 | SOAP     |       nan |               | final_validation_auc |

