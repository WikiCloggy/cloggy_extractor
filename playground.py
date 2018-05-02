from dogDetector import DogDetector
import numpy as np
import matplotlib.pyplot as plt
import imageProcessor
import cv2

dd = DogDetector()

img = cv2.imread('./images/dog_lay1.jpg')

result = dd.detectsOneDog(img)
print(result)

#dog1__.jpg
#rect = (28 + 25, 2 + 25, 517 - 50, 447 - 50)
#rect = (28, 2, 517, 422)
#dog3.jpg
#rect = (113, 34, 325, 305)
#sample_dog.jpg
#rect = (111, 189, 236, 375)

rect = dd.getDogRect(result, img)
print("Rect is %d, %d, %d, %d" %rect)

dogAreaImage = imageProcessor.drawRectangle(img, rect)
dogAreaImage = cv2.cvtColor(dogAreaImage, cv2.COLOR_BGR2RGB)
plt.title('Dog Area')
plt.imshow(dogAreaImage)
plt.show()

mask = imageProcessor.deleteBackground(img, rect)

plt.title('result')
plt.imshow(mask, 'gray')
plt.show()

cv2.imwrite('result.png', mask * 255)