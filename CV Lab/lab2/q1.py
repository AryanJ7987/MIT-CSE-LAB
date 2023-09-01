import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale
img = cv2.imread('C:\\Users\\ugcse.PG-CP.000\\Downloads\\highres.jpg', cv2.IMREAD_GRAYSCALE)

# Perform histogram equalization
equ = cv2.equalizeHist(img)

# Calculate histograms
hist_original, bins_original = np.histogram(img.flatten(), 256, [0, 256])
hist_equalized, bins_equalized = np.histogram(equ.flatten(), 256, [0, 256])

# Display original image, equalized image, and histograms
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(equ, cmap='gray')
plt.title('Equalized Image')

plt.subplot(2, 2, 3)
plt.hist(img.flatten(), 256, [0, 256], color='b', alpha=0.5, label='Original')
plt.hist(equ.flatten(), 256, [0, 256], color='r', alpha=0.5, label='Equalized')
plt.legend()
plt.title('Histograms')

plt.tight_layout()
plt.show()
