import cv2
from face_antispoofing.antispoofing import antispoof
from face_antispoofing.detection import face_detector, cascade_detection
from models import resnet_antispoof

vid = cv2.VideoCapture(0)
model = antispoof.Antispoof(resnet_antispoof.ResnetAntispoof())
detector = face_detector.FaceDetector(cascade_detection.CascadeDetector())

while True:
    ret, frame = vid.read()

    try: 
        faces = detector.detect_face(frame)

        for x, y, w, h in faces:
            prediction = model.model_predict(frame[x: x+w][y: y+w])
            if prediction:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
    except:
        pass

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
