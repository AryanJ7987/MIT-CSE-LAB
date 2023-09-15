import cv2

# Load two images
image1 = cv2.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962018\\resource\\et1.jfif", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962018\\resource\\et2.jfif", cv2.IMREAD_GRAYSCALE)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect and compute SIFT keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Brute-force matcher
bf = cv2.BFMatcher()

# Match descriptors using KNN (K-Nearest Neighbors)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw the matches
matching_result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)

# Display the result
cv2.imshow("SIFT Matches", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
