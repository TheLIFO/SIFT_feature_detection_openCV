import cv2
import numpy as np

video = cv2.VideoCapture(0)
check, frame2 = video.read()
orb = cv2.ORB_create(nfeatures=500)


gray1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
kpts1, des1  = orb.detectAndCompute(gray1, None)
gray1 = cv2.drawKeypoints (gray1, kpts1, None)

while True:        
    check, frame2 = video.read()
    cv2.imshow("live", frame2)
    cv2.waitKey(10)

    if not check: 
        break
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    

    kpts2, des2  = orb.detectAndCompute(gray2, None)
    gray2 = cv2.drawKeypoints (gray2, kpts2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if (des1 is None) | (des2 is None):
        continue
    matches = bf.match (des1, des2)
    matches = sorted (matches, key = lambda x:x.distance)

    matches = [m for m in matches if m.distance<60]
    matching_result = cv2.drawMatches(gray1, kpts1, gray2, kpts2, matches[:10], None, flags=2)
    cv2.imshow("Results", matching_result)

    
    key = cv2.waitKey(10)
    if key == ord('q'):
       break
    
    if key == ord('n'):
        gray1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        kpts1, des1  = orb.detectAndCompute(gray1, None)
        gray1 = cv2.drawKeypoints (gray1, kpts1, None)

video.release()
cv2.destroyAllWindows()