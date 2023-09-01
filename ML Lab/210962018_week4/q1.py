import cv2

def create_binary_image(image_path, threshold_value):
    # Load the image in grayscale
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding
    _, binary_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_BINARY)

    return binary_image

def main():
    image_path = "C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962018\\img.jfif"
    threshold_value = 128  # You can adjust this threshold value as needed

    binary_image = create_binary_image(image_path, threshold_value)

    cv2.imshow('Original Image', cv2.imread(image_path))
    cv2.imshow('Binary Image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
