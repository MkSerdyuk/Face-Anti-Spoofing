import torch
from torchvision import transforms as T
import torchvision
import torch.nn as nn


class Antispoof:

    def __init__(self, model_path):
        self.model = torchvision.models.mobilenet_v3_large()
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 2),
        )
        self.model.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')
            )
            )
        self.model.eval()

    def model_predict(self, frame):
        transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=(224, 224)),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        frame = transform(frame)
        with torch.no_grad():
            logits = self.model(frame.unsqueeze(0))
            return torch.argmax(logits, dim=1).cpu().numpy()
