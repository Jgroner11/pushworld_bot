import torch
import torch.nn as nn
import torch.nn.functional as F

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