import numpy as np
import cv2


def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('frame', frame)
 
    #if event == cv2.EVENT_MOUSEMOVE:
        # displaying the coordinates
        # on the Shell
    #    print(x, ' ', y)

    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = frame[y, x, 0]
        g = frame[y, x, 1]
        r = frame[y, x, 2]
        cv2.putText(frame, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('frame', frame)
  
vid_path = "./handball-detection/output.mp4"
cap = cv2.VideoCapture(vid_path)
seen = 0
while(cap.isOpened()):
  ret, frame = cap.read()
  if seen==34:
    expand_x = 0
    expand_y = 0
    xy1 = (632-expand_x,216-expand_y)
    xy2 = (676+expand_x,256+expand_y)
    cv2.rectangle(frame, xy1, xy2, (255,0,0), 2)
    cv2.imshow('frame',frame)
    cv2.setMouseCallback('frame', click_event)

    if cv2.waitKey(0) & 0xFF == ord('q'):
      break
  seen +=1

cap.release()
cv2.destroyAllWindows()


