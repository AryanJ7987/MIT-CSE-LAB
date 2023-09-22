import cv2
import numpy as np

def detect_keypoints(image, num_octaves, num_scales):
    # Create a list to store keypoints
    keypoints = []

    # Generate Gaussian pyramids
    pyramids = []
    for octave in range(num_octaves):
        pyramid = [image]
        for scale in range(1, num_scales + 3):
            # Calculate sigma based on scale and octave
            sigma = 1.6 * 2 ** (scale / num_scales)  # Adjust this formula as needed
            blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
            pyramid.append(blurred_image)
        pyramids.append(pyramid)

    # Compute the Difference of Gaussians (DoG) pyramid
    dog_pyramids = []
    for octave in range(num_octaves):
        dog_octave = [pyramids[octave][i + 1] - pyramids[octave][i] for i in range(num_scales + 2)]
        dog_pyramids.append(dog_octave)

    # Find keypoints in the DoG pyramids
    for octave in range(num_octaves):
        for scale in range(1, num_scales + 1):
            for y in range(1, image.shape[0] - 1):
                for x in range(1, image.shape[1] - 1):
                    # Compare the pixel value with its 26 neighbors in the current and adjacent scales
                    pixel_value = dog_pyramids[octave][scale][y, x]
                    is_maxima = True
                    is_minima = True
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            for ds in [-1, 0, 1]:
                                if dy == 0 and dx == 0 and ds == 0:
                                    continue
                                neighbor_value = dog_pyramids[octave][scale + ds][y + dy, x + dx]
                                if pixel_value < neighbor_value:
                                    is_maxima = False
                                if pixel_value > neighbor_value:
                                    is_minima = False
                    if is_maxima or is_minima:
                        keypoints.append((x, y, octave, scale))

    return keypoints

# Load an image (replace 'your_image.jpg' with your image file)
image = cv2.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962018\\resource\\et1.jfif", cv2.IMREAD_GRAYSCALE)

# Specify the number of octaves and scales per octave
num_octaves = 4
num_scales = 3

# Detect keypoints
keypoints = detect_keypoints(image, num_octaves, num_scales)

# Draw keypoints on the original image
image_with_keypoints = cv2.drawKeypoints(image, [cv2.KeyPoint(x, y, 1) for x, y, _, _ in keypoints], outImage=None)
cv2.imshow('Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
