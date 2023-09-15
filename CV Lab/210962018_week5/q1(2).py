import cv2

# Load an image from file
image = cv2.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962018\\resource\\sudoku.png", cv2.IMREAD_GRAYSCALE)

# Initialize the FAST detector with default parameters
fast = cv2.FastFeatureDetector_create()

# Detect keypoints (corners) in the image
keypoints = fast.detect(image, None)

# Draw keypoints on the original image
output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

# Display the image with keypoints
cv2.imshow("FAST Keypoints", output_image)

# Wait for a key press and close the window
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
