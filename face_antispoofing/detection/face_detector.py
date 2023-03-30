class FaceDetector:
    def __init__(self, detector):
        self.detector = detector

    def detect_face(self, frame):
        return self.detector.detect_face(frame)
