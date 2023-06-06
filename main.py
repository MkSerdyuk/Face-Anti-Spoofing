import cv2
import numpy as np
from face_antispoofing.antispoofing import antispoof
from face_antispoofing.detection import face_detector, mtcnn_detection
from models import mobilenet_antispoof, resnet_antispoof

vid = cv2.VideoCapture(0)
model = antispoof.Antispoof(mobilenet_antispoof.MobileNetAntispoof())
detector = face_detector.FaceDetector(mtcnn_detection.MTCNNDetector())
PADDING = 10

FRAMES_PER_CALCULATION = 5  # How often the result is recalculated

frame_counter = 0
logits_sum = np.array([[0, 0]], dtype=float)
prediction_sum = -1

while True:
    ret, frame = vid.read()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    try:
        faces = detector.detect_face(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for x0, y0, x1, y1 in faces:
            x0 -= PADDING
            y0 -= 3 * PADDING
            x1 += PADDING
            y1 += PADDING

            prediction, logits = model.model_predict(frame_rgb[x0:x1][y0:y1])

            # if cv2.waitKey(1) & 0xFF == ord(" "):
            #    prediction_sum = -1
            #    frame_counter = 1

            if FRAMES_PER_CALCULATION > 1:
                frame_counter += 1
                logits_sum += logits
                logits = logits_sum / frame_counter

                if FRAMES_PER_CALCULATION == frame_counter:
                    prediction_sum = np.argmax(logits_sum, axis=1)
                    logits_sum = np.array([[0, 0]], dtype=float)
                    frame_counter = 1

                prediction = prediction_sum

            print(prediction)

            if prediction:
                if prediction != -1:
                    color = (0, 255, 0)
                    prediction = "REAL"
                else:
                    color = (0, 255, 255)
                    prediction = "LOADING..."
            else:
                color = (0, 0, 255)
                prediction = "SPOOF"

            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

            (w, h), _ = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            cv2.rectangle(frame, (x0, y0 - h), (x0 + w, y0), color, -1)
            cv2.putText(frame, prediction, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            (w, h), _ = cv2.getTextSize("real", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(frame, f"real {logits[0][1]}", (x0, y1 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"spoof {logits[0][0]}", (x0, y1 + 2 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    except:
        pass

    cv2.imshow("frame", frame)


vid.release()
cv2.destroyAllWindows()
