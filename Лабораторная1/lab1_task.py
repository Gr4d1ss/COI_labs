# -*- coding: utf-8 -*-
"""
Лабораторная работа 1. Задание: работа с матрицами в NumPy.

Задание:
1. Сгенерировать две матрицы средствами NumPy
2. Вычислить средние значения в каждой строке каждой матрицы
3. К каждому значению строки первой матрицы прибавить среднее значение
   соответствующей строки второй матрицы
4. Проделать аналогично со второй матрицей
"""

import numpy as np

try:
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "all"
except ImportError:
    pass

# Генерируем две матрицы 3x4 со случайными числами от 0 до 10
rnd = np.random.default_rng(seed=42)
matrix1 = rnd.integers(0, 10, (3, 4))
matrix2 = rnd.integers(0, 10, (3, 4))

print("Матрица 1:")
print(matrix1)
print("\nМатрица 2:")
print(matrix2)

# Средние значения по строкам
mean1 = np.mean(matrix1, axis=1)
mean2 = np.mean(matrix2, axis=1)

print("\nСредние по строкам матрицы 1:", mean1)
print("Средние по строкам матрицы 2:", mean2)

# К каждой строке matrix1 прибавляем среднее соответствующей строки matrix2 и наоборот
# reshape(-1, 1) нужен для broadcasting
result1 = matrix1 + mean2.reshape(-1, 1)
result2 = matrix2 + mean1.reshape(-1, 1)

print("\nРезультат 1 (матрица 1 + средние строк матрицы 2):")
print(result1)
print("\nРезультат 2 (матрица 2 + средние строк матрицы 1):")
print(result2)

# Проверка для первой строки
print("\nПроверка:")
print(f"  Строка 1 матрицы 1: {matrix1[0]} + среднее строки 1 матрицы 2 = {mean2[0]:.2f}")
print(f"  Результат: {result1[0]}")
print(f"  Строка 1 матрицы 2: {matrix2[0]} + среднее строки 1 матрицы 1 = {mean1[0]:.2f}")
print(f"  Результат: {result2[0]}")
