import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("D:/DATA Backup 14.04.25/Files/Computer Vision/flower.jpeg")

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a Kernel and Apply Erosion
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded_image = cv2.erode(gray_image, kernel, iterations=1)

# Display
plt.imshow(eroded_image, cmap='gray')
plt.title("Eroded Image")
plt.axis('off')
plt.show()
