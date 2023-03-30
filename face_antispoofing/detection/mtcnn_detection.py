from facenet_pytorch import MTCNN


class MTCNNDetector:
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True)

    def detect_face(self, frame):
        faces, _ = self.mtcnn.detect(frame)
        faces = [map(int, face) for face in faces]
        return faces
