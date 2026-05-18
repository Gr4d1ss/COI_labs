# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import sys
sys.path.append('../utility')
from segmentation_utils import region_growingHSV, QTree

image = cv.imread('mandarin.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

print(f"Image shape: {image.shape}")

plt.figure(figsize=(20, 4))
plt.subplot(1, 6, 1)
plt.imshow(image_rgb)
plt.title('1. Original')
plt.axis('off')

h, w = image.shape[:2]
seeds = [(h//3, w//3), (h//2, w//2), (h//3*2, w//4)]
threshold = 80
segmented_region = region_growingHSV(image_hsv, seeds, threshold)
result_region = cv.bitwise_and(image, image, mask=segmented_region)
plt.subplot(1, 6, 2)
plt.imshow(cv.cvtColor(result_region, cv.COLOR_BGR2RGB))
plt.title('2. Region Growing')
plt.axis('off')

qt = QTree(stdThreshold=0.3, minPixelSize=8, img=image.copy())
qt.subdivide()
tree_image = qt.render_img(thickness=1, color=(0, 0, 0))
plt.subplot(1, 6, 3)
plt.imshow(cv.cvtColor(tree_image, cv.COLOR_BGR2RGB))
plt.title('3. QuadTree')
plt.axis('off')

binary_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
distance_map = ndimage.distance_transform_edt(binary_image)
local_max = peak_local_max(distance_map, min_distance=20, labels=binary_image)
peaks_mask = np.zeros_like(distance_map, dtype=bool)
peaks_mask[tuple(local_max.T)] = True
markers = ndimage.label(peaks_mask, structure=np.ones((3, 3)))[0]
labels = watershed(-distance_map, markers, mask=binary_image)
plt.subplot(1, 6, 4)
plt.imshow(labels, cmap='nipy_spectral')
plt.title('4. Watershed')
plt.axis('off')

pixels = gray.reshape(-1, 1)
K = 4
kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
labels_kmeans = kmeans.fit_predict(pixels)
segments_kmeans = np.uint8(kmeans.cluster_centers_[labels_kmeans].reshape(gray.shape))
plt.subplot(1, 6, 5)
plt.imshow(segments_kmeans, cmap='gray')
plt.title('5. K-means')
plt.axis('off')

print("\nRunning Mean Shift...")
flat_image = image.reshape((-1, 3))
flat_image = np.float32(flat_image)
bandwidth = estimate_bandwidth(flat_image, quantile=0.08, n_samples=3000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled = ms.labels_
segments_ms = np.unique(labeled)
print(f'Mean Shift segments: {segments_ms.shape[0]}')

total = np.zeros((segments_ms.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total / count
avg = np.uint8(avg)
mean_shift_image = avg[labeled].reshape((image.shape))
plt.subplot(1, 6, 6)
plt.imshow(mean_shift_image)
plt.title(f'6. Mean Shift\n({segments_ms.shape[0]} segments)')
plt.axis('off')
plt.tight_layout()
plt.savefig('lab6_segmentation_methods.png', dpi=150)
plt.show()

lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])
mask_orange = cv.inRange(image_hsv, lower_orange, upper_orange)
result_orange = cv.bitwise_and(image, image, mask=mask_orange)

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(image_rgb)
plt.title('Original')
plt.axis('off')
plt.subplot(1, 4, 2)
plt.imshow(mask_orange, cmap='gray')
plt.title('Orange Mask')
plt.axis('off')
plt.subplot(1, 4, 3)
plt.imshow(cv.cvtColor(result_orange, cv.COLOR_BGR2RGB))
plt.title('Segmented (HSV)')
plt.axis('off')

h_channel = image_hsv[:, :, 0]
plt.subplot(1, 4, 4)
plt.hist(h_channel.flatten(), bins=180, range=[0, 180], color='orange', alpha=0.7)
plt.axvline(10, color='r', linestyle='--', label='Lower')
plt.axvline(25, color='b', linestyle='--', label='Upper')
plt.title('Hue Histogram')
plt.xlabel('Hue')
plt.ylabel('Pixel Count')
plt.legend()
plt.tight_layout()
plt.savefig('lab6_orange_mask.png', dpi=150)
plt.show()

print("\nLab 6 completed!")
print(f"Orange mask pixels: {np.sum(mask_orange > 0)}")
