import cv2

img_path = r"C:\Users\ugcse.PG-CP.000\Desktop\210962018\l31.jfif"

# Read the image using OpenCV
img = cv2.imread(img_path)

# Apply Box filter
box_filtered = cv2.boxFilter(img, -1, (3, 3))

# Apply Gaussian filter
gaussian_filtered = cv2.GaussianBlur(img, (3, 3), 0)

# Display the original, box filtered, and Gaussian filtered images side by side
combined = cv2.hconcat([img, box_filtered, gaussian_filtered])
cv2.imshow("Comparison: Original | Box Filtered | Gaussian Filtered", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
