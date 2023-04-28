import cv2
from face_antispoofing.antispoofing import antispoof
from face_antispoofing.detection import face_detector, mtcnn_detection
from models import mobilenet_antispoof

vid = cv2.VideoCapture(0)
model = antispoof.Antispoof(mobilenet_antispoof.MobileNetAntispoof())
detector = face_detector.FaceDetector(mtcnn_detection.MTCNNDetector())
PADDING = 25

while True:
    ret, frame = vid.read()

    try:
        faces = detector.detect_face(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for x0, y0, x1, y1 in faces:
            x0 -= PADDING
            y0 -= PADDING
            x1 += PADDING
            y1 += PADDING
            prediction = model.model_predict(frame_rgb[x0:x1][y0:y1])
            if not prediction:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                continue
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
    except:
        pass

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
