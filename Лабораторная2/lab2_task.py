# -*- coding: utf-8 -*-
"""
Лабораторная работа 2. Задание: реализация эквализации гистограммы.

Задание:
Реализовать адаптивную эквализацию (выравнивание) гистограммы самостоятельно
по формуле:
    LUT[i] = 255 * (sum(Hist[1..i])) / (sum(Hist[1..255]))
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Настройка для отображения полного интерактивного вывода
try:
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "all"
except ImportError:
    pass

# Изменим стандартный размер графиков matplotlib
plt.rcParams["figure.figsize"] = [6, 4]

# ============================================================
# Шаг 1: Загрузка изображений
# ============================================================
print("=" * 60)
print("ШАГ 1: Загрузка изображений")
print("=" * 60)

import os

# Устанавливаем рабочую директорию как папку лабораторной
os.chdir(os.path.dirname(os.path.abspath(__file__)))

image1 = cv.imread('lenna.png')
image2 = cv.imread('winter_cat.png')

rgb_image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
rgb_image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray_image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

print(f"Изображение 1 (lenna): {gray_image1.shape}")
print(f"Изображение 2 (winter_cat): {gray_image2.shape}")

# ============================================================
# Шаг 2: Реализация эквализации гистограммы
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 2: Реализация эквализации гистограммы")
print("=" * 60)


def equalize_histogram(image):
    """
    Реализация эквализации гистограммы по формуле:
        LUT[i] = 255 * (sum(Hist[1..i])) / (sum(Hist[1..255]))
    
    Parameters:
    -----------
    image : np.ndarray
        Входное изображение в градациях серого (8-битное)
    
    Returns:
    --------
    np.ndarray
        Эквализованное изображение
    """
    # Шаг 2.1: Вычисляем гистограмму изображения
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()  # Преобразуем в одномерный массив
    
    # Шаг 2.2: Вычисляем кумулятивную сумму (накопленную сумму)
    # sum(Hist[1..i]) для каждого i
    cumsum = np.cumsum(hist)
    
    # Шаг 2.3: Вычисляем общую сумму пикселей
    # sum(Hist[1..255]) — это общее количество пикселей
    total_pixels = cumsum[-1]
    
    # Шаг 2.4: Создаём LUT (Look-Up Table) по формуле
    # LUT[i] = 255 * cumsum[i] / total_pixels
    lut = np.round(255.0 * cumsum / total_pixels).astype(np.uint8)
    
    # Шаг 2.5: Применяем LUT к изображению
    # Im[i,j] = LUT[Im[i,j]]
    equalized_image = lut[image]
    
    return equalized_image, hist, lut


# Применяем эквализацию к обоим изображениям
equalized_gray1, hist1, lut1 = equalize_histogram(gray_image1)
equalized_gray2, hist2, lut2 = equalize_histogram(gray_image2)

print("Эквализация выполнена успешно!")
print(f"LUT для изображения 1: первые 10 значений = {lut1[:10]}")
print(f"LUT для изображения 1: последние 10 значений = {lut1[-10:]}")

# ============================================================
# Шаг 3: Сравнение с встроенной функцией OpenCV
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 3: Сравнение с cv.equalizeHist()")
print("=" * 60)

# Встроенная эквализация OpenCV
opencv_equalized1 = cv.equalizeHist(gray_image1)
opencv_equalized2 = cv.equalizeHist(gray_image2)

# Проверяем совпадение результатов
match1 = np.array_equal(equalized_gray1, opencv_equalized1)
match2 = np.array_equal(equalized_gray2, opencv_equalized2)

print(f"Совпадение с OpenCV (lenna): {match1}")
print(f"Совпадение с OpenCV (winter_cat): {match2}")

if match1 and match2:
    print("\n✅ Результаты полностью совпадают с cv.equalizeHist()!")
else:
    # Если есть небольшие различия, покажем статистику
    diff1 = np.abs(equalized_gray1.astype(int) - opencv_equalized1.astype(int))
    diff2 = np.abs(equalized_gray2.astype(int) - opencv_equalized2.astype(int))
    print(f"\n⚠️ Максимальное различие (lenna): {np.max(diff1)}")
    print(f"⚠️ Максимальное различие (winter_cat): {np.max(diff2)}")

# ============================================================
# Шаг 4: Визуализация результатов
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 4: Визуализация результатов")
print("=" * 60)

range_hist = [0, 256]

# --- Изображение 1 (Lenna) ---
gs = plt.GridSpec(3, 3, figure=plt.figure(figsize=(15, 12)))

# Оригинальное изображение
plt.subplot(gs[0])
plt.imshow(gray_image1, cmap='gray')
plt.title('Оригинал (Lenna)')
plt.axis('off')

# Эквализованное изображение (наша реализация)
plt.subplot(gs[1])
plt.imshow(equalized_gray1, cmap='gray')
plt.title('Эквализация (наша реализация)')
plt.axis('off')

# Эквализованное изображение (OpenCV)
plt.subplot(gs[2])
plt.imshow(opencv_equalized1, cmap='gray')
plt.title('Эквализация (OpenCV)')
plt.axis('off')

# Гистограмма оригинала
plt.subplot(gs[3])
plt.hist(gray_image1.reshape(-1), 256, range_hist)
plt.title('Гистограмма оригинала')
plt.xlabel('Яркость')
plt.ylabel('Количество пикселей')

# Гистограмма эквализованного (наша реализация)
plt.subplot(gs[4])
plt.hist(equalized_gray1.reshape(-1), 256, range_hist)
plt.title('Гистограмма после эквализации')
plt.xlabel('Яркость')
plt.ylabel('Количество пикселей')

# Гистограмма эквализованного (OpenCV)
plt.subplot(gs[5])
plt.hist(opencv_equalized1.reshape(-1), 256, range_hist)
plt.title('Гистограмма после эквализации (OpenCV)')
plt.xlabel('Яркость')
plt.ylabel('Количество пикселей')

# Функция LUT
plt.subplot(gs[6])
plt.plot(lut1)
plt.title('Функция LUT (наша реализация)')
plt.xlabel('Входная яркость')
plt.ylabel('Выходная яркость')
plt.grid(True)

# Кумулятивная гистограмма
plt.subplot(gs[7])
cumsum1 = np.cumsum(hist1)
plt.plot(cumsum1 / cumsum1[-1] * 255)
plt.title('Кумулятивная гистограмма')
plt.xlabel('Яркость')
plt.ylabel('Накопленная сумма')
plt.grid(True)

# Разница между нашей реализацией и OpenCV
plt.subplot(gs[8])
diff = np.abs(equalized_gray1.astype(int) - opencv_equalized1.astype(int))
plt.imshow(diff, cmap='hot')
plt.title('Разница (наша vs OpenCV)')
plt.colorbar(label='Абсолютная разница')
plt.axis('off')

plt.tight_layout()
plt.savefig('equalization_result_lenna.png', dpi=150, bbox_inches='tight')
print("График сохранён: equalization_result_lenna.png")
plt.show()

# --- Изображение 2 (Winter Cat) ---
gs2 = plt.GridSpec(2, 2, figure=plt.figure(figsize=(12, 10)))

plt.subplot(gs2[0])
plt.imshow(gray_image2, cmap='gray')
plt.title('Оригинал (Winter Cat)')
plt.axis('off')

plt.subplot(gs2[1])
plt.imshow(equalized_gray2, cmap='gray')
plt.title('После эквализации')
plt.axis('off')

plt.subplot(gs2[2])
plt.hist(gray_image2.reshape(-1), 256, range_hist)
plt.title('Гистограмма оригинала')
plt.xlabel('Яркость')
plt.ylabel('Количество пикселей')

plt.subplot(gs2[3])
plt.hist(equalized_gray2.reshape(-1), 256, range_hist)
plt.title('Гистограмма после эквализации')
plt.xlabel('Яркость')
plt.ylabel('Количество пикселей')

plt.tight_layout()
plt.savefig('equalization_result_cat.png', dpi=150, bbox_inches='tight')
print("График сохранён: equalization_result_cat.png")
plt.show()

# ============================================================
# Шаг 5: Статистика
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 5: Статистика")
print("=" * 60)

print(f"\nИзображение 1 (Lenna):")
print(f"  Оригинал:  min={np.min(gray_image1):3d}, max={np.max(gray_image1):3d}, "
      f"mean={np.mean(gray_image1):6.2f}, std={np.std(gray_image1):6.2f}")
print(f"  Эквализация: min={np.min(equalized_gray1):3d}, max={np.max(equalized_gray1):3d}, "
      f"mean={np.mean(equalized_gray1):6.2f}, std={np.std(equalized_gray1):6.2f}")

print(f"\nИзображение 2 (Winter Cat):")
print(f"  Оригинал:  min={np.min(gray_image2):3d}, max={np.max(gray_image2):3d}, "
      f"mean={np.mean(gray_image2):6.2f}, std={np.std(gray_image2):6.2f}")
print(f"  Эквализация: min={np.min(equalized_gray2):3d}, max={np.max(equalized_gray2):3d}, "
      f"mean={np.mean(equalized_gray2):6.2f}, std={np.std(equalized_gray2):6.2f}")

print("\n" + "=" * 60)
print("ЗАДАНИЕ ВЫПОЛНЕНО УСПЕШНО!")
print("=" * 60)
