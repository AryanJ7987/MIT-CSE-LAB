import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962018\\resource\\haris.jpg", 0)  # Replace 'your_image.jpg' with your image file

# Define Harris corner detection parameters
block_size = 3   # Neighborhood size for computing the Harris matrix (changed to an odd value)
ksize = 3        # Aperture parameter for Sobel gradient calculation
k = 0.04         # Harris corner detection constant
threshold = 0.01  # Threshold for corner response

# Calculate gradients using Sobel operators
Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize)
Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize)

# Compute elements of the Harris matrix
A = cv2.GaussianBlur(Ix * Ix, (block_size, block_size), 0)
B = cv2.GaussianBlur(Ix * Iy, (block_size, block_size), 0)
C = cv2.GaussianBlur(Iy * Iy, (block_size, block_size), 0)

# Calculate the corner response function
det_H = A * C - B**2
trace_H = A + C
R = det_H - k * (trace_H**2)

# Apply a threshold to the corner response values
corners = np.where(R > threshold * R.max())

# Draw circles around detected corners
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for y, x in zip(*corners):
    cv2.circle(output_image, (x, y), 5, (0, 0, 255), -1)  # Draw a red circle

# Display the result
cv2.imshow('Harris Corners', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
