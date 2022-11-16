# Higher-order Neural Additive Models (HONAM)

## Requirements

We have implemented the code in the following python environment:

* python 3.8.12
* pytorch 1.10.2
* pandas 1.2.3
* numpy 1.21.2
* scikit-learn 1.0.2

## Quick Start

We provide an example code for the California Housing Prices dataset.

For training:
```shell
python run.py --mode=train --dataset=clifornia_housing
```

## Using HONAM in Your Code

### Sklearn interface

HONAM supports sklearn-style interface.

```python
from model import HONAM

model = HONAM(...)
model.fit(x_train, y_train)
prediction = model.predict(x_test) 
```

## Citation

```
@article{kim2022higher,
  title = {Higher-order Neural Additive Models: An Interpretable Machine Learning Model with Feature Interactions},
  author = {Kim, Minkyu and Choi, Hyun-Soo and Kim, Jinho},
  journal = {arXiv preprint arXiv:2209.15409},
  year = {2022}
}
```
