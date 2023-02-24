from facenet_pytorch import InceptionResnetV1


class MTCNNDetector:

    resnet = InceptionResnetV1(preatrained='vggface2').eval()

    def detect_face(self, frame):
        return self.resnet(frame.unsqueeze(0))
