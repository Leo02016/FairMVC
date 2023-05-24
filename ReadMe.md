# Required packages
* PyTorch
* kmeans_pytorch
* sklearn


# Commands
Experiment on Credit Card Data Set with non-contrastive reguarlization.
```
python main.py -d credit -mode ncontra
```

Experiment on Credit Card Data Set with contrastive reguarlization.
```
python main.py -d credit -mode contra -alpha 5
```

## Flags:
*-d: data set
*-g: the index of the gpu
*-hid: the hidden feature dimension
*-alpha: the coefficient of fairness constraint
*-beta: the coefficient of contrastive or non-contrastive regularization
*-gamma: the coefficient of KL divergence loss
*-miss: whether to enable missing feature scenario
*-purturbed: whether to enable noisy feature scenario
*-mode: use contrastive or non-contrastive regularizatio (default: non-contrastive regularization)

# Reference
@inproceedings{zheng2023fairness,
  title={Fairness-aware Multi-view Clustering},
  author={Zheng, Lecheng and Zhu, Yada and He, Jingrui},
  booktitle={Proceedings of the 2023 SIAM International Conference on Data Mining (SDM)},
  pages={856--864},
  year={2023},
  organization={SIAM}
}
