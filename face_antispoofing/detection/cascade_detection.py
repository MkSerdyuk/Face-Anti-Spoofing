import cv2


class CascadeDetector:
    def detect_face(self, frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        faces = [(x, y, x + w, y + h) for x, y, w, h in faces]
        return faces
