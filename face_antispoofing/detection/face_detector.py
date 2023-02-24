from cascade_detection import CascadeDetector


class FaceDetector:

    detector = CascadeDetector  # по умолчанию стоит детекция каскадами Хаара

    def detect_face(self, frame):
        return self.detector.detect_face(frame)
