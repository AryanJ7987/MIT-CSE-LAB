import numpy as np
import cv2

img = cv2.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962018\\l31.jfif")

cv2.imshow("original",img)

#blur the image using gaussian blur
blurred_img = cv2.GaussianBlur(img, (5, 5),0)

cv2.imshow("gaussian",blurred_img)

#subtract blurred image from original, then add to original

var1 = cv2.subtract(img,blurred_img)
img_unsharp=cv2.add(img,var1)

#output unsharp image
cv2.imshow("unsharp_image",img_unsharp/255)
cv2.waitKey(0)
cv2.destroyAllWindows()