import numpy as np
import cv2



cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read(0)
    cv2.rectangle(frame, (100,40), (200,150), (0,255,0), 3)
    cv2.putText(frame, "class_name", (110,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()