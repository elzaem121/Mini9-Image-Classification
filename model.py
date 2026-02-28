import os
import numpy as np
import torch
import torch.nn as nn


# ================================
#   Residual Block
# ================================
class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()

        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        # Shortcut projection if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.dropout:
            out = self.dropout(out)

        out += self.shortcut(x)
        return self.relu(out)


# ================================
#   Mini9 ResNet Model
# ================================
class Mini9ResNet(nn.Module):
    """
    Custom lightweight ResNet-style model used during training.
    """
    def __init__(self, num_classes=9):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout=0.1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=0.2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout=0.3)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout=0.4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout):
        layers = [ResidualBlock(in_channels, out_channels, stride, dropout)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ================================
#   Competition Model Wrapper
# ================================
class Model:
    """
    This class is the interface used by Codabench.
    It loads the pretrained model weights and exposes a `predict()` method.
    """

    def __init__(self):
        # Target categories (order must match training)
        self.categories = [
            "airplane", "automobile", "bird", "cat",
            "deer", "dog", "horse", "ship", "truck"
        ]

        # Always run on CPU during Codabench evaluation
        self.device = torch.device("cpu")

        # Normalization statistics (from training)
        self.mean = np.array([0.52461963, 0.55828897, 0.58782098], dtype=np.float32)
        self.std = np.array([0.18552486, 0.18374406, 0.19512308], dtype=np.float32)

        # Build model architecture
        self.model = Mini9ResNet(num_classes=9)

        # Load weights from same directory as this file
        weights_path = os.path.join(os.path.dirname(__file__), "best_model.pth")

        state_dict = torch.load(weights_path, map_location=self.device)

        # Support both cases:
        # 1) direct state_dict
        # 2) checkpoint["model_state_dict"]
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            self.model.load_state_dict(state_dict["model_state_dict"])
        else:
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    # ================================
    #   PREDICTION METHOD
    # ================================
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions for a batch of test images.

        :param x_test: numpy array of shape (N, 32, 32, 3)
        :return: numpy array of labels as strings (N,)
        """
        predictions = []
        batch_size = 64
        n = x_test.shape[0]

        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = x_test[i:i+batch_size]

                # Preprocessing: float → normalize → torch tensor
                processed = []
                for img in batch:
                    img = img.astype(np.float32) / 255.0
                    img = (img - self.mean) / self.std
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                    processed.append(img_tensor)

                batch_tensor = torch.stack(processed).to(self.device)

                # Inference
                outputs = self.model(batch_tensor)
                _, preds = outputs.max(1)

                # Convert indices → labels
                for idx in preds.cpu().numpy():
                    predictions.append(self.categories[idx])

        return np.array(predictions)
