import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("D:/DATA Backup 14.04.25/Files/Computer Vision/flower.jpeg")

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny Edge Detection
edges = cv2.Canny(gray_image, 100, 200)

# Display
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.show()
