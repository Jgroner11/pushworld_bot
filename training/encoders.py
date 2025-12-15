import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


class IntEncoder(nn.Module):
    def __init__(self, input_shape, max_size=torch.inf):
        super(IntEncoder, self).__init__()
        self._size = 0
        self._map = {}
        self.input_shape = input_shape
        self.max_size = max_size

    def classify(self, x):
        if x.shape == self.input_shape:
            return self.encode_single(x)
        elif x.shape[1:] == self.input_shape:
            return self.encode_multiple(x)
        else:
            raise Exception(f"Input doesn't match shape {self.input_shape}")

    def encode_single(self, x):
        key =  tuple(x.ravel().tolist())
        if key in self._map:
            return self._map[key]
        elif self._size == self.max_size:
            raise Exception("Encoder ran out of unique mappings")
        else:
            self._map[key] = self._size
            self._size += 1
            return self._map[key]

    def encode_multiple(self, arr):
        r = torch.tensor(dtype=torch.int64)
        for x in arr:
            r.append(self.encode(x))
        return r

class SimpleLinearEncoder(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        H, W, C = input_shape
        in_features = H * W * C
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = torch.as_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def classify(self, x):
        logits = self.forward(x)
        pred = torch.argmax(logits, dim=1)
        return pred.squeeze(0).item() if pred.ndim == 1 and pred.numel() == 1 else pred

class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        """
        input_shape: tuple (H, W, C)
        num_classes: number of output classes
        """
        super(SimpleCNN, self).__init__()

        # Unpack input shape (PyTorch expects channels first)
        H, W, C = input_shape
        in_channels = C
        assert in_channels == 3
        self.input_shape = input_shape

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # halve size

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # halve again
        )

        #  Trick: run dummy input through features to determine flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, H, W)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, num_classes)
        )

    def forward(self, x):
        if x.ndim == len(self.input_shape):
            x = x.unsqueeze(0)

        x = x.permute(0, 3, 1, 2) #conv layer requires (batch, C, H, V) but our model requires (batch, H, V, C)
        x = self.features(x)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def classify(self, x):
        logits = self.forward(torch.as_tensor(x))                     # shape: (batch_size, num_classes)
        predicted_classes = torch.argmax(logits, dim=1)  # get index of max logit per sample
        predicted_classes = predicted_classes.squeeze(0)
        if predicted_classes.ndim == 0:
            return predicted_classes.item()
        return predicted_classes

class VectorQuantizer(nn.Module):
    """
    Classical vector quantizer:
      image -> flatten -> PCA -> k-means -> observation id

    Must be fit(images) before classify().
    """
    def __init__(
        self,
        input_shape,
        num_classes,
        n_components=64,
        batch_size=4096,
        seed=0,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.n_components = n_components

        self._fitted = False

        self.pca = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
        )

        self.kmeans = KMeans(
            n_clusters=num_classes,
            random_state=seed,
            n_init="auto",
        )

    def _preprocess(self, x):
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 3:
            x = x[None, ...]

        assert x.shape[1:] == self.input_shape, \
            f"Expected {self.input_shape}, got {x.shape[1:]}"

        if x.max() > 1.0:
            x = x / 255.0

        return x.reshape(x.shape[0], -1)

    def fit(self, images):
        X = self._preprocess(images)

        bs = self.pca.batch_size
        for i in range(0, X.shape[0], bs):
            self.pca.partial_fit(X[i:i + bs])

        Z = self.pca.transform(X)
        with threadpool_limits(limits=1, user_api="openmp"):
            self.kmeans.fit(Z)

        self._fitted = True
        return self

    def classify(self, x):
        if not self._fitted:
            raise RuntimeError(
                "VQEncoder must be fit(images) before classify()."
            )

        X = self._preprocess(x)
        Z = self.pca.transform(X)
        ids = self.kmeans.predict(Z)

        if np.asarray(x).ndim == 3:
            return int(ids[0])
        return ids

