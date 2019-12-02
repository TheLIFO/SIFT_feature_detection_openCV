import cv2
import numpy as np
from matplotlib import pyplot as plt

#gray1 = cv2.imread ("Origami.jpg", cv2.IMREAD_GRAYSCALE)

video = cv2.VideoCapture(0)
check, frame2 = video.read()
orb = cv2.ORB_create(nfeatures=500)


gray1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
kpts1, des1  = orb.detectAndCompute(gray1, None)
gray1 = cv2.drawKeypoints (gray1, kpts1, None)

gray1_canny = cv2.Canny(gray1,threshold1=100,threshold2=200)
#plt.subplot(121),plt.imshow(gray1,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(gray1_canny,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#plt.show()

while True:    
    check, frame2 = video.read()
    if not check: 
        break
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Capturing",gray2)

    kpts2, des2  = orb.detectAndCompute(gray2, None)
    gray2 = cv2.drawKeypoints (gray2, kpts2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if (des1 is None) | (des2 is None):
        continue
    matches = bf.match (des1, des2)
    matches = sorted (matches, key = lambda x:x.distance)

    matches = [m for m in matches if m.distance<60]
    matching_result = cv2.drawMatches(gray1, kpts1, gray2, kpts2, matches, None, flags=2)
    cv2.imshow("Results", matching_result)

    
    key = cv2.waitKey(1)
    if key == ord('q'):
       break
    
    if key == ord('n'):
        gray1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        kpts1, des1  = orb.detectAndCompute(gray1, None)
        gray1 = cv2.drawKeypoints (gray1, kpts1, None)

video.release()
cv2.destroyAllWindows()