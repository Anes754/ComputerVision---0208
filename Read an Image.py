import cv2
import matplotlib.pyplot as plt

# Load and Display the Original Image
image = cv2.imread("D:/DATA Backup 14.04.25/Files/Computer Vision/flower.jpeg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()
