import cv2
import numpy as np 

drawing=False 

# mouse callback function
def interactive_drawing(event,x,y,flags,param):
    global ix,iy,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            
                cv2.line(img,(ix,iy),(x,y),(255,255,255),10)
                ix=x
                iy=y
                print x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False

           
    return x,y


    




img = np.zeros((512,512,3), np.float32)

cv2.namedWindow('image')
cv2.resizeWindow('image', 600,600)
cv2.setMouseCallback('image',interactive_drawing)
while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)&0xFF
    if k==27:
        cv2.imwrite("image.png",img)
        break
cv2.destroyAllWindows()