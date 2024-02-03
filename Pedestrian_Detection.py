import cv2
from imutils.object_detection import non_max_suppression
from imutils import resize
import numpy as np
import matplotlib.pyplot as plt

# بارگذاری مدل HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# بارگذاری عکس
img = cv2.imread('C:/Users/laptopiliya/Desktop/pedestrian-friendly-streets_0.jpg')
img = resize(img, height=500)

# شناسایی مردم
rects, weights = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)

# کشیدن مربع قبل از suppression
before_suppression = img.copy()
for x, y, w, h in rects:
    cv2.rectangle(before_suppression, (x, y), (x+w, y+h), (0, 0, 255), 2)

# کشیدن مربع بعد از suppression
plt.imshow(cv2.cvtColor(before_suppression, cv2.COLOR_BGR2RGB))
plt.title('Before Suppression')
plt.show()

r = np.array([[x, y, x+w, y+h] for x, y, w, h in rects])
pick = non_max_suppression(r, probs=None, overlapThresh=0.65)

# نمایش مربع بعد از suppression
after_suppression = img.copy()
for xa, ya, xb, yb in pick:
    cv2.rectangle(after_suppression, (xa, ya), (xb, yb), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(after_suppression, cv2.COLOR_BGR2RGB))
plt.title('After Suppression')
plt.show()
