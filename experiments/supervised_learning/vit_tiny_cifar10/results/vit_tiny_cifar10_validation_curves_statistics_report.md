# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 1.4317 | 0.0000 | 0.0000 | 1.4317 | 1.4317 |
| MILO_LW | 1.4308 | 0.0000 | 0.0000 | 1.4308 | 1.4308 |
| SGD | 1.6428 | 0.0000 | 0.0000 | 1.6428 | 1.6428 |
| ADAMW | 1.1809 | 0.0000 | 0.0000 | 1.1809 | 1.1809 |
| ADAGRAD | 1.7690 | 0.0000 | 0.0000 | 1.7690 | 1.7690 |
| ADEMAMIX | 1.2107 | 0.0000 | 0.0000 | 1.2107 | 1.2107 |
| SOAP | 0.9016 | 0.0000 | 0.0000 | 0.9016 | 0.9016 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  1.43171 | 1.43077  | MILO_LW  |       nan |               | final_validation_loss |
| MILO          | SGD           |  1.43171 | 1.64279  | MILO     |       nan |               | final_validation_loss |
| MILO          | ADAMW         |  1.43171 | 1.18086  | ADAMW    |       nan |               | final_validation_loss |
| MILO          | ADAGRAD       |  1.43171 | 1.76898  | MILO     |       nan |               | final_validation_loss |
| MILO          | ADEMAMIX      |  1.43171 | 1.21072  | ADEMAMIX |       nan |               | final_validation_loss |
| MILO          | SOAP          |  1.43171 | 0.901609 | SOAP     |       nan |               | final_validation_loss |
| MILO_LW       | SGD           |  1.43077 | 1.64279  | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADAMW         |  1.43077 | 1.18086  | ADAMW    |       nan |               | final_validation_loss |
| MILO_LW       | ADAGRAD       |  1.43077 | 1.76898  | MILO_LW  |       nan |               | final_validation_loss |
| MILO_LW       | ADEMAMIX      |  1.43077 | 1.21072  | ADEMAMIX |       nan |               | final_validation_loss |
| MILO_LW       | SOAP          |  1.43077 | 0.901609 | SOAP     |       nan |               | final_validation_loss |
| SGD           | ADAMW         |  1.64279 | 1.18086  | ADAMW    |       nan |               | final_validation_loss |
| SGD           | ADAGRAD       |  1.64279 | 1.76898  | SGD      |       nan |               | final_validation_loss |
| SGD           | ADEMAMIX      |  1.64279 | 1.21072  | ADEMAMIX |       nan |               | final_validation_loss |
| SGD           | SOAP          |  1.64279 | 0.901609 | SOAP     |       nan |               | final_validation_loss |
| ADAMW         | ADAGRAD       |  1.18086 | 1.76898  | ADAMW    |       nan |               | final_validation_loss |
| ADAMW         | ADEMAMIX      |  1.18086 | 1.21072  | ADAMW    |       nan |               | final_validation_loss |
| ADAMW         | SOAP          |  1.18086 | 0.901609 | SOAP     |       nan |               | final_validation_loss |
| ADAGRAD       | ADEMAMIX      |  1.76898 | 1.21072  | ADEMAMIX |       nan |               | final_validation_loss |
| ADAGRAD       | SOAP          |  1.76898 | 0.901609 | SOAP     |       nan |               | final_validation_loss |
| ADEMAMIX      | SOAP          |  1.21072 | 0.901609 | SOAP     |       nan |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 49.2000 | 0.0000 | 0.0000 | 49.2000 | 49.2000 |
| MILO_LW | 48.9630 | 0.0000 | 0.0000 | 48.9630 | 48.9630 |
| SGD | 39.4667 | 0.0000 | 0.0000 | 39.4667 | 39.4667 |
| ADAMW | 57.4519 | 0.0000 | 0.0000 | 57.4519 | 57.4519 |
| ADAGRAD | 34.6667 | 0.0000 | 0.0000 | 34.6667 | 34.6667 |
| ADEMAMIX | 57.0519 | 0.0000 | 0.0000 | 57.0519 | 57.0519 |
| SOAP | 68.8889 | 0.0000 | 0.0000 | 68.8889 | 68.8889 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  49.2    |  48.963  | MILO     |       nan |               | final_validation_accuracy |
| MILO          | SGD           |  49.2    |  39.4667 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADAMW         |  49.2    |  57.4519 | ADAMW    |       nan |               | final_validation_accuracy |
| MILO          | ADAGRAD       |  49.2    |  34.6667 | MILO     |       nan |               | final_validation_accuracy |
| MILO          | ADEMAMIX      |  49.2    |  57.0519 | ADEMAMIX |       nan |               | final_validation_accuracy |
| MILO          | SOAP          |  49.2    |  68.8889 | SOAP     |       nan |               | final_validation_accuracy |
| MILO_LW       | SGD           |  48.963  |  39.4667 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  48.963  |  57.4519 | ADAMW    |       nan |               | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  48.963  |  34.6667 | MILO_LW  |       nan |               | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  48.963  |  57.0519 | ADEMAMIX |       nan |               | final_validation_accuracy |
| MILO_LW       | SOAP          |  48.963  |  68.8889 | SOAP     |       nan |               | final_validation_accuracy |
| SGD           | ADAMW         |  39.4667 |  57.4519 | ADAMW    |       nan |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  39.4667 |  34.6667 | SGD      |       nan |               | final_validation_accuracy |
| SGD           | ADEMAMIX      |  39.4667 |  57.0519 | ADEMAMIX |       nan |               | final_validation_accuracy |
| SGD           | SOAP          |  39.4667 |  68.8889 | SOAP     |       nan |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  57.4519 |  34.6667 | ADAMW    |       nan |               | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  57.4519 |  57.0519 | ADAMW    |       nan |               | final_validation_accuracy |
| ADAMW         | SOAP          |  57.4519 |  68.8889 | SOAP     |       nan |               | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  34.6667 |  57.0519 | ADEMAMIX |       nan |               | final_validation_accuracy |
| ADAGRAD       | SOAP          |  34.6667 |  68.8889 | SOAP     |       nan |               | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  57.0519 |  68.8889 | SOAP     |       nan |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.4849 | 0.0000 | 0.0000 | 0.4849 | 0.4849 |
| MILO_LW | 0.4894 | 0.0000 | 0.0000 | 0.4894 | 0.4894 |
| SGD | 0.3875 | 0.0000 | 0.0000 | 0.3875 | 0.3875 |
| ADAMW | 0.5640 | 0.0000 | 0.0000 | 0.5640 | 0.5640 |
| ADAGRAD | 0.3378 | 0.0000 | 0.0000 | 0.3378 | 0.3378 |
| ADEMAMIX | 0.5625 | 0.0000 | 0.0000 | 0.5625 | 0.5625 |
| SOAP | 0.6896 | 0.0000 | 0.0000 | 0.6896 | 0.6896 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.484866 | 0.489408 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO          | SGD           | 0.484866 | 0.387529 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.484866 | 0.564032 | ADAMW    |       nan |               | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.484866 | 0.337832 | MILO     |       nan |               | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.484866 | 0.562504 | ADEMAMIX |       nan |               | final_validation_f1_score |
| MILO          | SOAP          | 0.484866 | 0.689622 | SOAP     |       nan |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.489408 | 0.387529 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.489408 | 0.564032 | ADAMW    |       nan |               | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.489408 | 0.337832 | MILO_LW  |       nan |               | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.489408 | 0.562504 | ADEMAMIX |       nan |               | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.489408 | 0.689622 | SOAP     |       nan |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.387529 | 0.564032 | ADAMW    |       nan |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.387529 | 0.337832 | SGD      |       nan |               | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.387529 | 0.562504 | ADEMAMIX |       nan |               | final_validation_f1_score |
| SGD           | SOAP          | 0.387529 | 0.689622 | SOAP     |       nan |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.564032 | 0.337832 | ADAMW    |       nan |               | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.564032 | 0.562504 | ADAMW    |       nan |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.564032 | 0.689622 | SOAP     |       nan |               | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.337832 | 0.562504 | ADEMAMIX |       nan |               | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.337832 | 0.689622 | SOAP     |       nan |               | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.562504 | 0.689622 | SOAP     |       nan |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 1

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.8808 | 0.0000 | 0.0000 | 0.8808 | 0.8808 |
| MILO_LW | 0.8812 | 0.0000 | 0.0000 | 0.8812 | 0.8812 |
| SGD | 0.8377 | 0.0000 | 0.0000 | 0.8377 | 0.8377 |
| ADAMW | 0.9221 | 0.0000 | 0.0000 | 0.9221 | 0.9221 |
| ADAGRAD | 0.8093 | 0.0000 | 0.0000 | 0.8093 | 0.8093 |
| ADEMAMIX | 0.9139 | 0.0000 | 0.0000 | 0.9139 | 0.9139 |
| SOAP | 0.9527 | 0.0000 | 0.0000 | 0.9527 | 0.9527 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|----------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.880849 | 0.881228 | MILO_LW  |       nan |               | final_validation_auc |
| MILO          | SGD           | 0.880849 | 0.837719 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADAMW         | 0.880849 | 0.922077 | ADAMW    |       nan |               | final_validation_auc |
| MILO          | ADAGRAD       | 0.880849 | 0.809305 | MILO     |       nan |               | final_validation_auc |
| MILO          | ADEMAMIX      | 0.880849 | 0.913928 | ADEMAMIX |       nan |               | final_validation_auc |
| MILO          | SOAP          | 0.880849 | 0.952745 | SOAP     |       nan |               | final_validation_auc |
| MILO_LW       | SGD           | 0.881228 | 0.837719 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.881228 | 0.922077 | ADAMW    |       nan |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.881228 | 0.809305 | MILO_LW  |       nan |               | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.881228 | 0.913928 | ADEMAMIX |       nan |               | final_validation_auc |
| MILO_LW       | SOAP          | 0.881228 | 0.952745 | SOAP     |       nan |               | final_validation_auc |
| SGD           | ADAMW         | 0.837719 | 0.922077 | ADAMW    |       nan |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.837719 | 0.809305 | SGD      |       nan |               | final_validation_auc |
| SGD           | ADEMAMIX      | 0.837719 | 0.913928 | ADEMAMIX |       nan |               | final_validation_auc |
| SGD           | SOAP          | 0.837719 | 0.952745 | SOAP     |       nan |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.922077 | 0.809305 | ADAMW    |       nan |               | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.922077 | 0.913928 | ADAMW    |       nan |               | final_validation_auc |
| ADAMW         | SOAP          | 0.922077 | 0.952745 | SOAP     |       nan |               | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.809305 | 0.913928 | ADEMAMIX |       nan |               | final_validation_auc |
| ADAGRAD       | SOAP          | 0.809305 | 0.952745 | SOAP     |       nan |               | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.913928 | 0.952745 | SOAP     |       nan |               | final_validation_auc |

