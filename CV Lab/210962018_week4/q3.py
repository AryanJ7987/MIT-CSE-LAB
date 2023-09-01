import cv2
image=cv2.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962018\\traffic1.jpg")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lb = (0, 100, 100)
ud = (10, 255, 255)
#red color
# h_min = 0
# h_max = 10
# s_min = 100
# s_max = 255
# v_min = 100
# v_max = 255


mask = cv2.inRange(hsv_image, lb, ud)
segmented_image = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('segmented_image.jpg', segmented_image)