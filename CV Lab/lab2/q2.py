import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the sample image and the reference image in grayscale
sample_img = cv2.imread('C:\\Users\\ugcse.PG-CP.000\\Downloads\\highres.jpg', cv2.IMREAD_GRAYSCALE)
reference_img = cv2.imread('C:\\Users\\ugcse.PG-CP.000\\Downloads\\res.png', cv2.IMREAD_GRAYSCALE)

# Perform histogram equalization on the sample image and reference image
sample_equ = cv2.equalizeHist(sample_img)
reference_equ = cv2.equalizeHist(reference_img)

# Calculate histograms for the sample images
hist_sample_original, bins_sample_original = np.histogram(sample_img.flatten(), 256, [0, 256])
hist_sample_equalized, bins_sample_equalized = np.histogram(sample_equ.flatten(), 256, [0, 256])

# Calculate histograms for the reference images
hist_reference_original, bins_reference_original = np.histogram(reference_img.flatten(), 256, [0, 256])
hist_reference_equalized, bins_reference_equalized = np.histogram(reference_equ.flatten(), 256, [0, 256])

# Display original images, equalized images, and histograms
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(sample_equ, cmap='gray')
plt.title('Sample Equalized Image')

plt.subplot(2, 2, 2)
plt.imshow(reference_equ, cmap='gray')
plt.title('Reference Equalized Image')

plt.subplot(2, 2, 3)
plt.plot(hist_sample_original, color='b', alpha=0.5, label='Sample Original')
plt.plot(hist_sample_equalized, color='r', alpha=0.5, label='Sample Equalized')
plt.legend()
plt.title('Sample Image Histograms')

plt.subplot(2, 2, 4)
plt.plot(hist_reference_original, color='b', alpha=0.5, label='Reference Original')
plt.plot(hist_reference_equalized, color='r', alpha=0.5, label='Reference Equalized')
plt.legend()
plt.title('Reference Image Histograms')

plt.tight_layout()
plt.show()
