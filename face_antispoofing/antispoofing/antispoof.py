import torch
from torchvision import transforms as T


class Antispoof:

    def __init__(self, model):
        self.model = model
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
