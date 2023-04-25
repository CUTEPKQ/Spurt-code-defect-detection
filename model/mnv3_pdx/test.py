import paddlex as pdx
import matplotlib as plt
import cv2

model = pdx.load_model('output/mobilenetv3_small/best_model')
result = model.predict('data/defect/noisy_picture2.jpg')

print(result)

img = cv2.imread('data/defect/noisy_picture2.jpg')
cv2.imshow('img', img)
key = cv2.waitKey(0)