# Лабораторная работа 6. Сегментация изображений

## Цель работы
Изучить и применить различные методы сегментации изображений: разрастание областей, квадродерево, водораздел, кластеризация.

## Методы сегментации

### 1. Region Growing (Разрастание областей)

**Принцип работы:**
1. Задаются начальные точки ("семена") внутри объектов
2. Алгоритм проверяет соседние пиксели
3. Если пиксель похож на семя (по цвету) — добавляется в регион
4. Процесс продолжается рекурсивно

**Код из segmentation_utils.py:**
```python
def region_growingHSV(image_hsv, seeds, threshold):
    queue = deque([seed])
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            current_color = image_hsv[ny, nx]
            seed_color = image_hsv[seed_y, seed_x]
            diff = np.sqrt(np.sum((current_color - seed_color) ** 2))
            if diff < threshold:
                mask[ny, nx] = 255
                queue.append((nx, ny))
```

**Как работает:**
- `seeds` — координаты начальных точек на мандаринах
- `threshold` — порог похожести (евклидово расстояние в HSV)
- 4-связность — проверяются только соседи сверху/снизу/слева/справа
- `diff < threshold` — если цвет похож, пиксель добавляется в маску

**Применение:**
```python
seeds = [(h//3, w//3), (h//2, w//2), (h//3*2, w//4)]
segmented_region = region_growingHSV(image_hsv, seeds, threshold=80)
```

---

### 2. QuadTree (Квадродерево)

**Принцип работы:**
1. Изображение делится на 4 квадранта
2. Если дисперсия яркости в квадранте > порога — делится ещё на 4
3. Процесс продолжается рекурсивно пока не достигнут минимальный размер

**Код из segmentation_utils.py:**
```python
class QTree:
    def _subdivide_region(self, x, y, w, h):
        if w <= self.minPixelSize or h <= self.minPixelSize:
            self.is_leaf = True
            return
        
        region = self.img[y:y+h, x:x+w]
        std_val = np.std(gray_region)
        
        if std_val < self.stdThreshold * 255:
            self.is_leaf = True  # Однородная область — не делим
        else:
            # Делим на 4 квадранта
            half_w, half_h = w // 2, h // 2
            # Рекурсивный вызов для каждого квадранта
```

**Как работает:**
- `stdThreshold` — порог стандартного отклонения (0.3 = 30%)
- `minPixelSize` — минимальный размер блока (8 пикселей)
- `np.std()` — вычисляет неоднородность области
- Если область однородная — останавливаем деление

**Применение:**
```python
qt = QTree(stdThreshold=0.3, minPixelSize=8, img=image.copy())
qt.subdivide()
tree_image = qt.render_img(thickness=1, color=(0, 0, 0))
```

**Визуализация:**
- Рисуется прямоугольник вокруг каждого листа дерева
- Толщина=1 пиксель, цвет=чёрный

---

### 3. Watershed (Водораздел)

**Принцип работы:**
1. Изображение рассматривается как рельеф (яркость = высота)
2. "Вода" заполняет минимумы (тёмные области)
3. Где потоки воды встречаются — строятся границы (водоразделы)

**Код:**
```python
# 1. Бинаризация порогом Оцу
binary_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

# 2. Карта расстояний (distance transform)
distance_map = ndimage.distance_transform_edt(binary_image)

# 3. Локальные максимумы (центры объектов)
local_max = peak_local_max(distance_map, min_distance=20, labels=binary_image)

# 4. Маркеры для водораздела
markers = ndimage.label(peaks_mask, structure=np.ones((3, 3)))[0]

# 5. Применение watershed
labels = watershed(-distance_map, markers, mask=binary_image)
```

**Как работает:**
- `cv.THRESH_OTSU` — автоматически находит оптимальный порог бинаризации
- `distance_transform_edt()` — вычисляет расстояние от каждого пикселя до фона
- `peak_local_max()` — находит центры объектов (локальные максимумы расстояний)
- `watershed()` — "заполняет" бассейны от маркеров, строит границы

**Визуализация:**
- `cmap='nipy_spectral'` — каждая метка своим цветом

---

### 4. K-means Кластеризация

**Принцип работы:**
1. Все пиксели группируются в K кластеров
2. Центроиды кластеров инициализируются случайно
3. Итеративно: пиксели назначаются ближайшему центроиду, центроиды пересчитываются

**Код:**
```python
pixels = gray.reshape(-1, 1)  # Превращаем изображение в 1D массив
K = 4  # Количество кластеров

kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
labels_kmeans = kmeans.fit_predict(pixels)

# Восстанавливаем изображение из кластеров
segments_kmeans = np.uint8(kmeans.cluster_centers_[labels_kmeans].reshape(gray.shape))
```

**Как работает:**
- `gray.reshape(-1, 1)` — каждый пиксель становится точкой в 1D пространстве
- `n_clusters=4` — 4 уровня яркости (тёмный, средний, светлый, белый)
- `cluster_centers_` — средние значения яркости для каждого кластера
- Каждый пиксель заменяется значением центроида своего кластера

---

### 5. Mean Shift Кластеризация

**Принцип работы:**
1. Каждой точке ставится в соответствие окно (bandwidth)
2. Точка смещается к центру масс точек в окне
3. Процесс повторяется пока не сойдётся
4. Точки сошедшиеся в один центр — один кластер

**Код:**
```python
flat_image = image.reshape((-1, 3))  # Пиксели как точки в 3D (RGB)
flat_image = np.float32(flat_image)

bandwidth = estimate_bandwidth(flat_image, quantile=0.08, n_samples=3000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled = ms.labels_

# Вычисляем средний цвет для каждого кластера
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total / count
mean_shift_image = avg[labeled].reshape((image.shape))
```

**Как работает:**
- `quantile=0.08` — 8% точек для оценки bandwidth (размер окна)
- `bin_seeding=True` — ускоряет работу через предварительную группировку
- Не нужно задавать количество кластеров — определяется автоматически
- Каждый пиксель заменяется средним цветом своего кластера

---

### 6. Цветовая сегментация (HSV маска)

**Код:**
```python
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])
mask_orange = cv.inRange(hsv, lower_orange, upper_orange)
result_orange = cv.bitwise_and(image, image, mask=mask_orange)
```

**Как работает:**
- H ∈ [10, 25] — оранжевый оттенок (20-50° в реальном масштабе)
- S ∈ [100, 255] — высокая насыщенность
- V ∈ [100, 255] — достаточная яркость

---

## Сравнение методов

| Метод | Плюсы | Минусы |
|-------|-------|--------|
| **Region Growing** | Точная сегментация по цвету | Требует ручного выбора семян |
| **QuadTree** | Адаптивное разбиение | Не выделяет объекты, только области |
| **Watershed** | Автоматическое разделение объектов | Чувствителен к шуму, требует маркеров |
| **K-means** | Простой, быстрый | Нужно задавать K, только по яркости |
| **Mean Shift** | Не нужно задавать число кластеров | Медленный, много параметров |

## Параметры для мандаринов
```python
# Region Growing
seeds = [(h//3, w//3), (h//2, w//2), (h//3*2, w//4)]
threshold = 80

# QuadTree
stdThreshold = 0.3
minPixelSize = 8

# Watershed
min_distance = 20  # Минимальное расстояние между объектами

# K-means
K = 4  # 4 уровня яркости

# Mean Shift
quantile = 0.08  # 8% для bandwidth
```

## Запуск
```bash
cd Лабораторная6
python lab6_final.py
```

## Сохранённые результаты
- `lab6_segmentation_methods.png` — все 6 методов в одной картинке
- `lab6_orange_mask.png` — маска оранжевого цвета + гистограмма
