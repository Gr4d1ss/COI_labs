# -*- coding: utf-8 -*-
"""
Лабораторная работа 3. Цифровая обработка изображений.
Студент: Лебедев Константин, Группа: 24ВВИм2

Задания:
1. Image-68-1c19cc.jpg - улучшить методами из 2 и 3 лабораторных (инверсия, фильтры).
2. k23_s.jpg - удалить линии, чтобы текст можно было перевести.

"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Определяем директорию скрипта для корректных путей
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Настройки графиков
plt.rcParams["figure.figsize"] = [12, 8]
plt.ioff()

def getPSNR(I1, I2):
    s1 = cv.absdiff(I1, I2)
    s1 = np.float32(s1)
    s1 = s1 * s1
    sse = s1.sum()
    if sse <= 1e-10:
        return 0
    else:
        shape = I1.shape
        p = 1
        for i in shape:
            p *= i
        mse = 1.0 * sse / p
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr

def getSSIM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    I1 = np.float32(i1)
    I2 = np.float32(i2)
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv.GaussianBlur(I1*I1, (11, 11), 1.5) - mu1_2
    sigma2_2 = cv.GaussianBlur(I2*I2, (11, 11), 1.5) - mu2_2
    sigma12 = cv.GaussianBlur(I1*I2, (11, 11), 1.5) - mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2
    ssim_map = cv.divide(t3, t1)
    ssim = cv.mean(ssim_map)[:3]
    return ssim

# ЗАДАНИЕ 1: Улучшение Image-68-1c19cc.jpg

print("=" * 60)
print("ЗАДАНИЕ 1: Улучшение Image-68-1c19cc.jpg")
print("=" * 60)

img1_path = os.path.join(SCRIPT_DIR, 'Image-68-1c19cc.jpg')
img1 = cv.imdecode(np.fromfile(img1_path, dtype=np.uint8), cv.IMREAD_COLOR)
img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

# 1. Инверсия цветов (как в задании - негатив)
inverted = cv.bitwise_not(img1)
inverted_rgb = cv.cvtColor(inverted, cv.COLOR_BGR2RGB)

# 2. Переход в оттенки серого
gray = cv.cvtColor(inverted, cv.COLOR_BGR2GRAY)

# 3. Медианный фильтр (из раздела 2.2 примера) - убираем шум
median_5 = cv.medianBlur(gray, 5)

# 4. Повышение резкости (из раздела 3.2 примера)
# Ядро 1: центральное 9, остальные -1
kernel_sharp = np.asarray([[-1, -1, -1], 
                           [-1, 9, -1], 
                           [-1, -1, -1]])

result1 = cv.filter2D(median_5, -1, kernel_sharp)

# Визуализация
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(img1_rgb)
plt.title('Оригинал (негатив)')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(inverted_rgb)
plt.title('После инверсии')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(median_5, cmap='gray')
plt.title('Медианный фильтр 5x5')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(result1, cmap='gray')
plt.title('Итог (Медианный + Резкость)')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'step_task1.png'), dpi=100)
plt.close()

# Сохраняем результат
res1_path = os.path.join(SCRIPT_DIR, 'Image-68-1c19cc_result.jpg')
cv.imencode('.jpg', result1)[1].tofile(res1_path)
print("Сохранено: Image-68-1c19cc_result.jpg")

# ЗАДАНИЕ 2: Удаление линий с k23_s.jpg
# Метод: Частотная фильтрация (Фурье)

print("\n" + "=" * 60)
print("ЗАДАНИЕ 2: Удаление линий с k23_s.jpg (Метод Фурье)")
print("=" * 60)

img2_path = os.path.join(SCRIPT_DIR, 'k23_s.jpg')
img2 = cv.imdecode(np.fromfile(img2_path, dtype=np.uint8), cv.IMREAD_COLOR)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# 1. Прямое преобразование Фурье
dft = np.fft.fft2(img2_gray)
dft_shift = np.fft.fftshift(dft)

# 2. Создаем маску для удаления горизонтальных линий
# Горизонтальные линии = вертикальная полоса в центре спектра Фурье.
# Мы её обнуляем.
rows, cols = img2_gray.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols), np.uint8)
w_band = 4  # Ширина полосы удаления
mask[:, ccol-w_band:ccol+w_band] = 0  # Обнуляем вертикальную полосу в центре

# 3. Применяем маску и обратное преобразование
dft_shift_masked = dft_shift * mask
f_ishift = np.fft.ifftshift(dft_shift_masked)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Нормализуем в диапазон 0-255
result_fourier = np.uint8(np.clip(img_back, 0, 255))

# 4. Бинаризация для улучшения читаемости
_, result2 = cv.threshold(result_fourier, 130, 255, cv.THRESH_BINARY)
result2 = cv.bitwise_not(result2)  # Инвертируем, чтобы текст был черным

# Визуализация спектра (для отчета)
magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
magnitude_masked = 20 * np.log(np.abs(dft_shift_masked) + 1)

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(img2_gray, cmap='gray')
plt.title('Оригинал с линиями')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Спектр Фурье (есть линии)')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(magnitude_masked, cmap='gray')
plt.title('Спектр после маски')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(result2, cmap='gray')
plt.title('Итог (Фурье + Бинаризация)')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'step_task2.png'), dpi=100)
plt.close()

res2_path = os.path.join(SCRIPT_DIR, 'k23_s_result.jpg')
res2_enh_path = os.path.join(SCRIPT_DIR, 'k23_s_result_enhanced.jpg')

cv.imencode('.jpg', result2)[1].tofile(res2_path)
cv.imencode('.jpg', result_fourier)[1].tofile(res2_enh_path)

print("Сохранено: k23_s_result.jpg (Читаемый текст)")
print("Сохранено: k23_s_result_enhanced.jpg (Мягкий вариант)")

print("\n" + "=" * 60)
print("Лабораторная работа 3 завершена!")
print("=" * 60)
