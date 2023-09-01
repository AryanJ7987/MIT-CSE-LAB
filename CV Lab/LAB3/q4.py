import cv2

img_path = r"C:\Users\ugcse.PG-CP.000\Desktop\210962018\l31.jfif"

# Read the image using OpenCV
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply Laplacian filter for edge detection
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Normalize Laplacian for display
normalized_laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Stack original image and Laplacian filtered image side by side
stacked_images = cv2.hconcat([img, normalized_laplacian])

# Display the stacked images
cv2.imshow("Original Image vs. Laplacian Filtered Image", stacked_images)
cv2.waitKey(0)
cv2.destroyAllWindows()
