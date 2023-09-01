import cv2
import numpy as np

img_path = r"C:\Users\ugcse.PG-CP.000\Desktop\210962018\l4.jpg"

# Read the image using OpenCV
img = cv2.imread(img_path)

# Calculate the gradient for each color channel using Sobel filters
gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the magnitude and direction of gradient for each channel
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y, gradient_x)

# Normalize the gradient magnitude for display
normalized_gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display the gradient magnitude
cv2.imshow("Gradient Magnitude", normalized_gradient_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()
