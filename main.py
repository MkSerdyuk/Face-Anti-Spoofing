import cv2
import face_capture

vid = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read()
    cv2.imshow('frame', face_capture.faceanalize(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()