import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("D:/DATA Backup 14.04.25/Files/Computer Vision/flower.jpeg")

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a Kernel and Apply Dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated_image = cv2.dilate(gray_image, kernel, iterations=1)

# Display
plt.imshow(dilated_image, cmap='gray')
plt.title("Dilated Image")
plt.axis('off')
plt.show()
