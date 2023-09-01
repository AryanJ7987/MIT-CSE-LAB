import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962018\\img.jfif")
pixels = image.reshape(-1, 3).astype(np.float32)

K = 10

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)

segmented_image = centers[labels.flatten()].reshape(image.shape)

plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()