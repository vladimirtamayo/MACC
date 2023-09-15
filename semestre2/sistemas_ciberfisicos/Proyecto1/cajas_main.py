import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

# Cargar la imagen desde el archivo
img1 = cv2.imread("images/IMG_20230827_174523.jpg")

# Convertir la imagen de BGR a RGB
img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# Mostrar la imagen en una figura
plt.figure()
plt.imshow(img_rgb)
plt.title("Caja 1")
plt.xticks([]), plt.yticks([])
plt.show()

# Convertir la imagen de BGR a escala de grises
img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Aplicar umbralización a la imagen en escala de grises
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

# Mostrar la imagen binarizada
plt.imshow(thresh) 
plt.show()

# Obtener el texto a partir de la imagen binarizada usando pytesseract
text = pytesseract.image_to_string(img_gray)

# Imprimir el texto obtenido por pytesseract
print("Tesseract: ", text)
