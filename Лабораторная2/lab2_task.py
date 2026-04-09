# -*- coding: utf-8 -*-
"""
Лабораторная работа 2. Задание: реализация эквализации гистограммы.

Формула:
    LUT[i] = 255 * (sum(Hist[1..i])) / (sum(Hist[1..255]))
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "all"
except ImportError:
    pass

plt.rcParams["figure.figsize"] = [6, 4]

# Загрузка изображений
image1 = cv.imread('lenna.png')
image2 = cv.imread('winter_cat.png')

gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray_image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

print(f"Изображение 1: {gray_image1.shape}")
print(f"Изображение 2: {gray_image2.shape}")


def equalize_histogram(image):
    """Эквализация гистограммы по формуле из задания."""
    hist = cv.calcHist([image], [0], None, [256], [0, 256]).flatten()
    cumsum = np.cumsum(hist)
    total = cumsum[-1]
    # LUT[i] = 255 * cumsum[i] / total
    lut = np.round(255.0 * cumsum / total).astype(np.uint8)
    return lut[image], hist, lut


# Применяем к обоим изображениям
eq1, hist1, lut1 = equalize_histogram(gray_image1)
eq2, hist2, lut2 = equalize_histogram(gray_image2)

# Сравнение с cv.equalizeHist
cv_eq1 = cv.equalizeHist(gray_image1)
cv_eq2 = cv.equalizeHist(gray_image2)

print(f"\nСовпадение с cv.equalizeHist (lenna): {np.array_equal(eq1, cv_eq1)}")
print(f"Совпадение с cv.equalizeHist (cat): {np.array_equal(eq2, cv_eq2)}")

# Визуализация
range_hist = [0, 256]

# Lenna
gs = plt.GridSpec(2, 2, figure=plt.figure(figsize=(12, 10)))
plt.subplot(gs[0, 0])
plt.imshow(gray_image1, cmap='gray')
plt.title('Оригинал')
plt.axis('off')

plt.subplot(gs[0, 1])
plt.imshow(eq1, cmap='gray')
plt.title('После эквализации')
plt.axis('off')

plt.subplot(gs[1, 0])
plt.hist(gray_image1.reshape(-1), 256, range=range_hist)
plt.title('Гистограмма оригинала')

plt.subplot(gs[1, 1])
plt.hist(eq1.reshape(-1), 256, range=range_hist)
plt.title('Гистограмма после эквализации')

plt.tight_layout()
plt.savefig('equalization_result_lenna.png', dpi=150, bbox_inches='tight')
plt.show()

# Winter cat
gs2 = plt.GridSpec(2, 2, figure=plt.figure(figsize=(12, 10)))
plt.subplot(gs2[0, 0])
plt.imshow(gray_image2, cmap='gray')
plt.title('Оригинал')
plt.axis('off')

plt.subplot(gs2[0, 1])
plt.imshow(eq2, cmap='gray')
plt.title('После эквализации')
plt.axis('off')

plt.subplot(gs2[1, 0])
plt.hist(gray_image2.reshape(-1), 256, range=range_hist)
plt.title('Гистограмма оригинала')

plt.subplot(gs2[1, 1])
plt.hist(eq2.reshape(-1), 256, range=range_hist)
plt.title('Гистограмма после эквализации')

plt.tight_layout()
plt.savefig('equalization_result_cat.png', dpi=150, bbox_inches='tight')
plt.show()

# Статистика
print(f"\nLenna:     ориг: min={gray_image1.min():3d}, max={gray_image1.max():3d}, "
      f"mean={gray_image1.mean():.1f}  =>  экв: min={eq1.min():3d}, max={eq1.max():3d}, "
      f"mean={eq1.mean():.1f}")
print(f"Cat:       ориг: min={gray_image2.min():3d}, max={gray_image2.max():3d}, "
      f"mean={gray_image2.mean():.1f}  =>  экв: min={eq2.min():3d}, max={eq2.max():3d}, "
      f"mean={eq2.mean():.1f}")
