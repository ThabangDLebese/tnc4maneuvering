# Temporal Neighborhood Coding for Maneuvering (TNC4Maneuvering)

<div style="text-align:center;">
    <img src="tnc4man_diagram.png" alt="tnc4maneuvering" width="400" height="350">
</div>

This repository implements the Temporal Neighborhood Coding for Maneuvering (TNC4Maneuvering) framework, which aims to understand maneuverability in smart transportation using acceleration datasets.


## Requirements

List of package dependencies required, along with their versions used for training and testing the model (see requirements.txt). To install them, you can run: ```python3 -m pip install -r requirements.txt```


## Code Usage

Below are descriptions of some of important parameters used throught the code:

| Parameter       | Description                                              |
| --------------- | -------------------------------------------------------- |
| `operational_day` | The dataset name, which can be one of `'one_ds'`, `'one_dl'`, or `'eight_d'`. |
| `initial_size` | Initial window size used in the window selection method. |
| `window_sizeT` | Terminal window size not to exceed in the window selection method. |
| `window_size` | Optimal window size used throughout the process. |
| `p_value` | Overall pre-selected p-value used throughout the process. |
| `no_of_cv` | Number of cross-validations (defaults to `1`). |
| `weight_t` | Debiasing weight (defaults to `0.05`). |
| `train_epochs` | Total number of training/testing epochs. |
|


**Window size selection**: For selecting a suitable window size, simply exercute:

```bash
python3 -m dataset.windowsize_selection --dataset <operational_day> --window_size0 <initial_size> --steps < step_size> --window_sizeT <terminal_size> --n_evals <no_evals>
```

**Data pre-processing**: This preprocessing script is useful for the three datasets. You can prepare your datasets by running:

```bash
python3 -m dataset.preprocessing --dataset <operational_day> --window_size <window_size> --p_value <pvalaue>
```

### Training

For training TNC4maneuvering encoder model, simply exercute: 

```bash
python3 -m tnc4maneuvering.tnc4maneuvering --data <operational_day> --train --cv <no_of_cv> --w <weight_t>
```

To evaluate downstream tasks of classification, clusterability, and multilinear regression, pre-pruning use:

```bash 
python3 -m evaluations.multlin_regression --dataset <operational_day> --window_size <window_size> --n_epochs <train_epochs> 
python3 -m evaluations.clustering --dataset <operational_day> --window_size <window_size>
python3 -m evaluations.classification --dataset <operational_day>
```

### Pruning methods

1. Further evaluation of downstream tasks with PCA pruning use:

```bash
python3 -m evaluations.prunedPCAmultlinreg --dataset <operational_day> --window_size <window_size> --n_epochs <train_epochs> 
python3 -m evaluations.prunedPCAcluster --dataset <operational_day> --window_size <window_size> 
python3 -m evaluations.prunedPCAclass --dataset <operational_day> --window_size <window_size> --cv <no_of_cv>
```

2. For further evaluation of downstream tasks with PCC pruning use:

```bash 
python3 -m evaluations.prunedPCCmultlinreg --dataset <operational_day> --window_size <window_size> --n_epochs <train_epochs> 
python3 -m evaluations.prunedPCCcluster --dataset <operational_day> --window_size <window_size>
python3 -m evaluations.prunedPCCclass --dataset <operational_day> --window_size <window_size> --cv <no_of_cv>
```

<!-- ## Contact
If you have questions, please create an issue or email tlebese@sigma-clermont.fr | This work adopted the TNC framework, based on papers
[1](https://arxiv.org/pdf/2106.00750), [2](https://ieeexplore.ieee.org/iel7/10159234/10159280/10159325.pdf?casa_token=YFXQY5R3grAAAAAA:FKNaWX5hElYeRG3Pfg_v28Heqpqsn_ZyGSjL3wfajzSoQ4c7c6pm_G45s9gOK97C38xc17Ym9_8). -->

## Acknowledgements
The implementation of TNC4Maneuvering relies on resources from the TNC framework, codebase and repository [1](https://arxiv.org/pdf/2106.00750), [2](https://ieeexplore.ieee.org/iel7/10159234/10159280/10159325.pdf?casa_token=YFXQY5R3grAAAAAA:FKNaWX5hElYeRG3Pfg_v28Heqpqsn_ZyGSjL3wfajzSoQ4c7c6pm_G45s9gOK97C38xc17Ym9_8). We thank the authors for open-sourcing their work. If you have questions, please create an issue.

