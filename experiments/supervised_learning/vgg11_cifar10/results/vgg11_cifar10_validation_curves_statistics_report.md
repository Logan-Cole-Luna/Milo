# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 1.1457 | 0.1074 | 0.0480 | 1.0123 | 1.2791 |
| MILO_LW | 1.1279 | 0.0375 | 0.0168 | 1.0813 | 1.1745 |
| SGD | 1.2217 | 0.0521 | 0.0233 | 1.1570 | 1.2864 |
| ADAMW | 0.7176 | 0.0676 | 0.0302 | 0.6336 | 0.8015 |
| ADAM_MINI | 1.9059 | 0.0448 | 0.0200 | 1.8502 | 1.9615 |
| NOVOGRAD | 1.3628 | 0.0601 | 0.0269 | 1.2882 | 1.4374 |
| ADAGRAD | 1.2200 | 0.0457 | 0.0204 | 1.1632 | 1.2768 |
| ADEMAMIX | 0.6959 | 0.0497 | 0.0222 | 0.6342 | 0.7575 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 1.1457   | 1.1279   | MILO_LW  | 0.740808    |               | final_validation_loss |
| MILO          | SGD           | 1.1457   | 1.22168  | MILO     | 0.2064      |               | final_validation_loss |
| MILO          | ADAMW         | 1.1457   | 0.717552 | ADAMW    | 0.000160213 | ***           | final_validation_loss |
| MILO          | ADAM_MINI     | 1.1457   | 1.90586  | MILO     | 1.62504e-05 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      | 1.1457   | 1.36282  | MILO     | 0.00692329  | **            | final_validation_loss |
| MILO          | ADAGRAD       | 1.1457   | 1.22     | MILO     | 0.209816    |               | final_validation_loss |
| MILO          | ADEMAMIX      | 1.1457   | 0.695852 | ADEMAMIX | 0.000202158 | ***           | final_validation_loss |
| MILO_LW       | SGD           | 1.1279   | 1.22168  | MILO_LW  | 0.013061    | *             | final_validation_loss |
| MILO_LW       | ADAMW         | 1.1279   | 0.717552 | ADAMW    | 1.60562e-05 | ***           | final_validation_loss |
| MILO_LW       | ADAM_MINI     | 1.1279   | 1.90586  | MILO_LW  | 2.78428e-09 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 1.1279   | 1.36282  | MILO_LW  | 0.00018135  | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 1.1279   | 1.22     | MILO_LW  | 0.00879649  | **            | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 1.1279   | 0.695852 | ADEMAMIX | 6.12238e-07 | ***           | final_validation_loss |
| SGD           | ADAMW         | 1.22168  | 0.717552 | ADAMW    | 1.80185e-06 | ***           | final_validation_loss |
| SGD           | ADAM_MINI     | 1.22168  | 1.90586  | SGD      | 2.3452e-08  | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 1.22168  | 1.36282  | SGD      | 0.00429283  | **            | final_validation_loss |
| SGD           | ADAGRAD       | 1.22168  | 1.22     | ADAGRAD  | 0.958183    |               | final_validation_loss |
| SGD           | ADEMAMIX      | 1.22168  | 0.695852 | ADEMAMIX | 2.03957e-07 | ***           | final_validation_loss |
| ADAMW         | ADAM_MINI     | 0.717552 | 1.90586  | ADAMW    | 7.12936e-09 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.717552 | 1.36282  | ADAMW    | 2.74791e-07 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.717552 | 1.22     | ADAMW    | 2.42595e-06 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.717552 | 0.695852 | ADEMAMIX | 0.580124    |               | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      | 1.90586  | 1.36282  | NOVOGRAD | 4.76942e-07 | ***           | final_validation_loss |
| ADAM_MINI     | ADAGRAD       | 1.90586  | 1.22     | ADAGRAD  | 9.86817e-09 | ***           | final_validation_loss |
| ADAM_MINI     | ADEMAMIX      | 1.90586  | 0.695852 | ADEMAMIX | 1.84654e-10 | ***           | final_validation_loss |
| NOVOGRAD      | ADAGRAD       | 1.36282  | 1.22     | ADAGRAD  | 0.00335391  | **            | final_validation_loss |
| NOVOGRAD      | ADEMAMIX      | 1.36282  | 0.695852 | ADEMAMIX | 8.67547e-08 | ***           | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 1.22     | 0.695852 | ADEMAMIX | 1.33081e-07 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 77.8578 | 1.1466 | 0.5128 | 76.4341 | 79.2815 |
| MILO_LW | 77.6415 | 0.6434 | 0.2877 | 76.8426 | 78.4404 |
| SGD | 67.4074 | 1.2785 | 0.5718 | 65.8199 | 68.9949 |
| ADAMW | 78.3022 | 1.4007 | 0.6264 | 76.5630 | 80.0414 |
| ADAM_MINI | 25.1467 | 2.6467 | 1.1837 | 21.8603 | 28.4330 |
| NOVOGRAD | 67.5733 | 0.9066 | 0.4054 | 66.4477 | 68.6990 |
| ADAGRAD | 54.6637 | 1.7333 | 0.7751 | 52.5116 | 56.8158 |
| ADEMAMIX | 77.5022 | 1.2558 | 0.5616 | 75.9429 | 79.0615 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  77.8578 |  77.6415 | MILO     | 0.725044    |               | final_validation_accuracy |
| MILO          | SGD           |  77.8578 |  67.4074 | MILO     | 9.12888e-07 | ***           | final_validation_accuracy |
| MILO          | ADAMW         |  77.8578 |  78.3022 | ADAMW    | 0.598552    |               | final_validation_accuracy |
| MILO          | ADAM_MINI     |  77.8578 |  25.1467 | MILO     | 5.43476e-08 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  77.8578 |  67.5733 | MILO     | 4.5329e-07  | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  77.8578 |  54.6637 | MILO     | 4.73441e-08 | ***           | final_validation_accuracy |
| MILO          | ADEMAMIX      |  77.8578 |  77.5022 | MILO     | 0.652691    |               | final_validation_accuracy |
| MILO_LW       | SGD           |  77.6415 |  67.4074 | MILO_LW  | 4.38281e-06 | ***           | final_validation_accuracy |
| MILO_LW       | ADAMW         |  77.6415 |  78.3022 | ADAMW    | 0.37723     |               | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  77.6415 |  25.1467 | MILO_LW  | 4.97759e-07 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  77.6415 |  67.5733 | MILO_LW  | 1.27093e-07 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  77.6415 |  54.6637 | MILO_LW  | 9.50017e-07 | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  77.6415 |  77.5022 | MILO_LW  | 0.832697    |               | final_validation_accuracy |
| SGD           | ADAMW         |  67.4074 |  78.3022 | ADAMW    | 1.37099e-06 | ***           | final_validation_accuracy |
| SGD           | ADAM_MINI     |  67.4074 |  25.1467 | SGD      | 9.90417e-08 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  67.4074 |  67.5733 | NOVOGRAD | 0.819458    |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  67.4074 |  54.6637 | SGD      | 2.14281e-06 | ***           | final_validation_accuracy |
| SGD           | ADEMAMIX      |  67.4074 |  77.5022 | ADEMAMIX | 1.48515e-06 | ***           | final_validation_accuracy |
| ADAMW         | ADAM_MINI     |  78.3022 |  25.1467 | ADAMW    | 1.42281e-08 | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  78.3022 |  67.5733 | ADAMW    | 2.27319e-06 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  78.3022 |  54.6637 | ADAMW    | 1.89227e-08 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  78.3022 |  77.5022 | ADAMW    | 0.369814    |               | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  25.1467 |  67.5733 | NOVOGRAD | 4.98072e-07 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  25.1467 |  54.6637 | ADAGRAD  | 1.72972e-07 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADEMAMIX      |  25.1467 |  77.5022 | ADEMAMIX | 3.24632e-08 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADAGRAD       |  67.5733 |  54.6637 | NOVOGRAD | 5.78361e-06 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADEMAMIX      |  67.5733 |  77.5022 | ADEMAMIX | 1.34055e-06 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  54.6637 |  77.5022 | ADEMAMIX | 3.44715e-08 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.7776 | 0.0110 | 0.0049 | 0.7640 | 0.7913 |
| MILO_LW | 0.7766 | 0.0067 | 0.0030 | 0.7683 | 0.7850 |
| SGD | 0.6703 | 0.0133 | 0.0060 | 0.6537 | 0.6868 |
| ADAMW | 0.7827 | 0.0131 | 0.0058 | 0.7665 | 0.7989 |
| ADAM_MINI | 0.1866 | 0.0327 | 0.0146 | 0.1461 | 0.2271 |
| NOVOGRAD | 0.6748 | 0.0090 | 0.0040 | 0.6636 | 0.6859 |
| ADAGRAD | 0.5412 | 0.0184 | 0.0082 | 0.5183 | 0.5641 |
| ADEMAMIX | 0.7758 | 0.0116 | 0.0052 | 0.7614 | 0.7903 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.777645 | 0.776627 | MILO     | 0.865368    |               | final_validation_f1_score |
| MILO          | SGD           | 0.777645 | 0.670275 | MILO     | 9.69233e-07 | ***           | final_validation_f1_score |
| MILO          | ADAMW         | 0.777645 | 0.782705 | ADAMW    | 0.526794    |               | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.777645 | 0.186605 | MILO     | 2.91318e-07 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.777645 | 0.674778 | MILO     | 3.22685e-07 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.777645 | 0.54122  | MILO     | 1.09148e-07 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.777645 | 0.775827 | MILO     | 0.806036    |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.776627 | 0.670275 | MILO_LW  | 4.35573e-06 | ***           | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.776627 | 0.782705 | ADAMW    | 0.390836    |               | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.776627 | 0.186605 | MILO_LW  | 1.01442e-06 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.776627 | 0.674778 | MILO_LW  | 9.11391e-08 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.776627 | 0.54122  | MILO_LW  | 1.20909e-06 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.776627 | 0.775827 | MILO_LW  | 0.898175    |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.670275 | 0.782705 | ADAMW    | 8.85119e-07 | ***           | final_validation_f1_score |
| SGD           | ADAM_MINI     | 0.670275 | 0.186605 | SGD      | 3.61008e-07 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.670275 | 0.674778 | NOVOGRAD | 0.550816    |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.670275 | 0.54122  | SGD      | 3.14068e-06 | ***           | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.670275 | 0.775827 | ADEMAMIX | 1.12102e-06 | ***           | final_validation_f1_score |
| ADAMW         | ADAM_MINI     | 0.782705 | 0.186605 | ADAMW    | 1.32569e-07 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.782705 | 0.674778 | ADAMW    | 1.11827e-06 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.782705 | 0.54122  | ADAMW    | 3.94669e-08 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.782705 | 0.775827 | ADAMW    | 0.405042    |               | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.186605 | 0.674778 | NOVOGRAD | 1.33886e-06 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.186605 | 0.54122  | ADAGRAD  | 4.23092e-07 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADEMAMIX      | 0.186605 | 0.775827 | ADEMAMIX | 2.38302e-07 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADAGRAD       | 0.674778 | 0.54122  | NOVOGRAD | 8.6771e-06  | ***           | final_validation_f1_score |
| NOVOGRAD      | ADEMAMIX      | 0.674778 | 0.775827 | ADEMAMIX | 5.91343e-07 | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.54122  | 0.775827 | ADEMAMIX | 8.53295e-08 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9642 | 0.0029 | 0.0013 | 0.9607 | 0.9678 |
| MILO_LW | 0.9649 | 0.0016 | 0.0007 | 0.9630 | 0.9669 |
| SGD | 0.9480 | 0.0028 | 0.0013 | 0.9445 | 0.9515 |
| ADAMW | 0.9717 | 0.0032 | 0.0014 | 0.9677 | 0.9756 |
| ADAM_MINI | 0.7777 | 0.0131 | 0.0059 | 0.7615 | 0.7940 |
| NOVOGRAD | 0.9456 | 0.0022 | 0.0010 | 0.9429 | 0.9483 |
| ADAGRAD | 0.9102 | 0.0069 | 0.0031 | 0.9016 | 0.9189 |
| ADEMAMIX | 0.9700 | 0.0035 | 0.0016 | 0.9655 | 0.9744 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.964232 | 0.964938 | MILO_LW  | 0.646502    |               | final_validation_auc |
| MILO          | SGD           | 0.964232 | 0.947997 | MILO     | 1.84613e-05 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.964232 | 0.971671 | ADAMW    | 0.00465838  | **            | final_validation_auc |
| MILO          | ADAM_MINI     | 0.964232 | 0.777748 | MILO     | 2.60654e-06 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.964232 | 0.945569 | MILO     | 4.8498e-06  | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.964232 | 0.910227 | MILO     | 1.02557e-05 | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.964232 | 0.969951 | ADEMAMIX | 0.0240088   | *             | final_validation_auc |
| MILO_LW       | SGD           | 0.964938 | 0.947997 | MILO_LW  | 1.71426e-05 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.964938 | 0.971671 | ADAMW    | 0.00558699  | **            | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.964938 | 0.777748 | MILO_LW  | 4.43766e-06 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.964938 | 0.945569 | MILO_LW  | 5.69721e-07 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.964938 | 0.910227 | MILO_LW  | 3.25504e-05 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.964938 | 0.969951 | ADEMAMIX | 0.0304837   | *             | final_validation_auc |
| SGD           | ADAMW         | 0.947997 | 0.971671 | ADAMW    | 1.78647e-06 | ***           | final_validation_auc |
| SGD           | ADAM_MINI     | 0.947997 | 0.777748 | SGD      | 3.93909e-06 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.947997 | 0.945569 | SGD      | 0.169932    |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.947997 | 0.910227 | SGD      | 6.7139e-05  | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.947997 | 0.969951 | ADEMAMIX | 6.74936e-06 | ***           | final_validation_auc |
| ADAMW         | ADAM_MINI     | 0.971671 | 0.777748 | ADAMW    | 1.848e-06   | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.971671 | 0.945569 | ADAMW    | 1.13188e-06 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.971671 | 0.910227 | ADAMW    | 3.64252e-06 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.971671 | 0.969951 | ADAMW    | 0.442119    |               | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.777748 | 0.945569 | NOVOGRAD | 5.65542e-06 | ***           | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.777748 | 0.910227 | ADAGRAD  | 8.78774e-07 | ***           | final_validation_auc |
| ADAM_MINI     | ADEMAMIX      | 0.777748 | 0.969951 | ADEMAMIX | 1.513e-06   | ***           | final_validation_auc |
| NOVOGRAD      | ADAGRAD       | 0.945569 | 0.910227 | NOVOGRAD | 0.000150547 | ***           | final_validation_auc |
| NOVOGRAD      | ADEMAMIX      | 0.945569 | 0.969951 | ADEMAMIX | 5.4088e-06  | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.910227 | 0.969951 | ADEMAMIX | 2.73537e-06 | ***           | final_validation_auc |

