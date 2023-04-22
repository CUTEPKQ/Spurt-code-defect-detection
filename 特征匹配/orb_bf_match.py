import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

MIN_MATCH_COUNT = 10  # 设置最低特征点匹配数量为10

e1 = cv.getTickCount()
img1 = cv.imread('test.jpg')
img2 = cv.imread('dst.jpg')
img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)


orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)


if len(matches) > MIN_MATCH_COUNT:
    # 获取关键点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # 计算变换矩阵和MASK

    M, mask = cv.estimateAffinePartial2D(src_pts, dst_pts, cv.RANSAC)
    matchesMask = mask.ravel().tolist()
else:
    print("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCH_COUNT))
    matchesMask = None

rows, cols = img2.shape[:2]
res = cv.warpAffine(img1, M, (cols,rows))

e2 = cv.getTickCount()
time = (e2 - e1) / cv.getTickFrequency()
print(time)
cv.imwrite('To_dst.jpg', res)
plt.subplot(121)
plt.imshow(img1, 'gray')
plt.subplot(122)
plt.imshow(res, 'gray')
plt.show()

