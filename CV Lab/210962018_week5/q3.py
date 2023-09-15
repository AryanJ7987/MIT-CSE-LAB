import cv2

# Load the pre-trained HOG detector for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load the test image in which you want to detect human objects
image = cv2.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962018\\resource\\et1.jfif", cv2.IMREAD_GRAYSCALE)

# Detect human objects in the image
found_locations, _ = hog.detectMultiScale(image)

# Draw rectangles around the detected human objects
for (x, y, w, h) in found_locations:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with detected human objects
cv2.imshow("Human Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
