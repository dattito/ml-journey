import dataclasses

import grain
import jax.numpy as jnp
import sklearn
import torch


class BreastCancerDataset(grain.sources.RandomAccessDataSource):
    def __init__(self, X, y):
        self.X = jnp.array(X)
        self.y = jnp.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return {"features": self.X[index], "label": self.y[index]}


@dataclasses.dataclass(frozen=True)
class NormalizeTransform(grain.transforms.Map):
    mean: float
    std: float

    def map(self, element):
        return {
            'features': (element['features'] - self.mean) / self.std,
            'label': element['label']  # Keep label unchanged
        }

data = sklearn.datasets.load_breast_cancer()
X_train_all, X_test, y_train_all, y_test = sklearn.model_selection.train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
    X_train_all, y_train_all, test_size=0.2, random_state=42
)

train_source = BreastCancerDataset(X_train, y_train)
val_source = BreastCancerDataset(X_val, y_val)
test_source = BreastCancerDataset(X_test, y_test)

train_loader = (
    grain.MapDataset.source(train_source)
    .map(NormalizeTransform(mean=0, std=0.25))
)
val_loader = (
    grain.MapDataset.source(val_source)
    .map(NormalizeTransform(mean=0, std=0.25))
)
test_loader = (
    grain.MapDataset.source(test_source)
    .map(NormalizeTransform(mean=0, std=0.25))
)
