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
          return torch.tensor([self.encode_single(x) for x in arr], dtype=torch.int64)


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

        if x.max() > 1.0:
            x = x / 255.0
        x = (x - 0.5) / 0.5   # optional but usually helps a lot

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

class HopfieldVQ(nn.Module):
    """
    Vector quantizer built on a modern (dense) Hopfield network.

    A modern Hopfield network stores K patterns X and retrieves the pattern
    closest to a query ξ by iterating:

        ξ_new = X · softmax(β · Xᵀ · ξ)

    At β → ∞ this collapses to hard nearest-neighbor lookup (standard VQ).
    At finite β it is a soft, differentiable VQ: the output is a
    temperature-weighted blend of all codewords, dominated by the nearest one.

    This class:
      1. Projects raw images to a low-dim feature vector (via a small CNN trunk).
      2. Runs Hopfield retrieval against a learned codebook of K patterns.
      3. Returns the index of the winning codeword (hard argmax over the
         softmax weights after convergence).

    The codebook is initialized via K-means on the features of a batch of
    images, then refined end-to-end with a commitment loss that pulls encoder
    outputs toward their assigned codeword.

    Args:
        input_shape: (H, W, C) of raw input images.
        num_classes:  K, the codebook size.
        feature_dim:  dimensionality of the feature space the Hopfield network
                      operates in.
        beta:         inverse temperature. Higher = sharper / harder retrieval.
        n_steps:      number of Hopfield update iterations per forward pass.
                      Convergence is typically reached in 1–3 steps.
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        feature_dim=64,
        beta=10.0,
        n_steps=3,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.beta = beta
        self.n_steps = n_steps
        self._fitted = False

        H, W, C = input_shape
        self.trunk = nn.Sequential(
            nn.Conv2d(C, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            flat_dim = self.trunk(dummy).shape[1]

        self.proj = nn.Linear(flat_dim, feature_dim)

        # Codebook: K stored patterns, each of dimension feature_dim.
        # Stored as (feature_dim, K) so Xᵀξ is a simple matmul.
        self.codebook = nn.Parameter(torch.randn(feature_dim, num_classes))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, x):
        """Raw images → L2-normalised feature vectors, shape (N, feature_dim)."""
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.max() > 1.0:
            x = x / 255.0
        x = x.permute(0, 3, 1, 2)           # (N, C, H, W)
        z = self.proj(self.trunk(x))         # (N, feature_dim)
        return F.normalize(z, dim=-1)

    def _hopfield_retrieve(self, xi):
        """
        Run Hopfield update iterations.

        xi:  (N, feature_dim) — query vectors (encoder outputs)
        X:   (feature_dim, K) — stored patterns (codebook), L2-normalised

        Each step:
            scores  = beta * X^T xi          shape (N, K)
            weights = softmax(scores, dim=1) shape (N, K)
            xi_new  = X @ weights^T          shape (N, feature_dim), then normalise

        Returns the final softmax weights (N, K), which peak at the nearest codeword.
        """
        X = F.normalize(self.codebook, dim=0)   # normalise stored patterns
        xi_cur = xi                              # (N, feature_dim)

        for _ in range(self.n_steps):
            scores  = self.beta * (xi_cur @ X)           # (N, K)
            weights = F.softmax(scores, dim=1)            # (N, K)
            xi_cur  = F.normalize(weights @ X.T, dim=-1) # (N, feature_dim)

        return weights  # (N, K)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Returns (weights, indices, commitment_loss).

        weights:         (N, K) soft assignment weights — differentiable.
        indices:         (N,)   hard argmax winner — for downstream use.
        commitment_loss: scalar — pulls encoder features toward their codeword.
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        xi = self._encode(x)                          # (N, feature_dim)
        weights = self._hopfield_retrieve(xi)         # (N, K)

        indices = weights.argmax(dim=1)               # (N,) hard assignment

        # Commitment loss: ||encoder_output - assigned_codeword||²
        X = F.normalize(self.codebook, dim=0)         # (feature_dim, K)
        assigned = X[:, indices].T                    # (N, feature_dim)
        commitment_loss = F.mse_loss(xi, assigned.detach())

        return weights, indices, commitment_loss

    def fit(self, images):
        """
        Initialise the codebook with K-means over encoder features.
        Call this once on the full dataset before training.
        """
        from sklearn.cluster import KMeans

        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(images, dtype=torch.float32)
            feats = self._encode(x).cpu().numpy()

        km = KMeans(n_clusters=self.num_classes, n_init="auto", random_state=0)
        km.fit(feats)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32).T  # (feature_dim, K)
        centers = F.normalize(centers, dim=0)
        self.codebook.data.copy_(centers)
        self._fitted = True
        return self

    def train_on_sequence(self, images, n_iters=100, batch_size=128, lr=3e-4, use_wandb=False):
        """
        Train purely on raw images using the VQ commitment loss — no CSCG involved.

        The commitment loss ||encoder_features - nearest_codeword||² pushes the
        encoder trunk to produce features that cluster tightly around codebook
        entries, while also pulling the codebook entries toward the data.
        Both the trunk/proj parameters and the codebook are updated together.
        """
        import wandb as _wandb

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        x = torch.as_tensor(np.asarray(images), dtype=torch.float32)
        N = x.shape[0]

        self.train()
        for step in range(n_iters):
            idx = torch.randperm(N)[:batch_size]
            _, _, commitment_loss = self.forward(x[idx])

            optimizer.zero_grad()
            commitment_loss.backward()
            optimizer.step()

            if use_wandb:
                _wandb.log({"hopfield_vq/commitment_loss": commitment_loss.item(),
                            "hopfield_vq/step": step})

        self.eval()

    @torch.no_grad()
    def classify(self, x):
        """Return hard codeword indices, shape (N,) or scalar."""
        x_in = np.asarray(x)
        single = x_in.ndim == 3
        _, indices, _ = self.forward(torch.as_tensor(x_in, dtype=torch.float32))
        indices = indices.cpu()
        return indices[0].item() if single else indices.numpy()