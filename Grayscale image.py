import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("D:/DATA Backup 14.04.25/Files/Computer Vision/flower.jpeg")  # Replace with your image path

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()
