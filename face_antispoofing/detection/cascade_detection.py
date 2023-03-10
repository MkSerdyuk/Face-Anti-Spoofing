import cv2


class CascadeDetector:

    def detect_face(self, frame):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return face_cascade.detectMultiScale(gray, 1.1, 4)
