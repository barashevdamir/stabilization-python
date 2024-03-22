import cv2
import numpy as np


def good_features_to_track(image, maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector=False,
                           harrisK=0.04):
    assert qualityLevel > 0 and minDistance >= 0 and maxCorners >= 0, "Неверные параметры функции"

    # Выбор метода обнаружения углов
    if useHarrisDetector:
        eig = cv2.cornerHarris(image, blockSize, 3, harrisK)
    else:
        eig = cv2.cornerMinEigenVal(image, blockSize, 3)
        # eig = corner_min_eigen_val(image, blockSize, 3)

    # Нормализация результата
    eig = cv2.normalize(eig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    eig = np.uint8(eig)

    # Порог для выбора сильных углов
    maxVal = np.max(eig) * qualityLevel
    _, eig_thresh = cv2.threshold(eig, maxVal, 255, cv2.THRESH_BINARY)

    # Поиск углов после применения порога
    corners = np.argwhere(eig_thresh > 0)
    corners = np.flip(corners, axis=1)  # Поменять местами X и Y для соответствия формату (x, y)

    # Фильтрация углов с использованием минимального расстояния
    if minDistance > 0:
        corners = cv2.dilate(np.float32(corners), None)
        final_corners = []
        for corner in corners:
            too_close = False
            for final_corner in final_corners:
                if np.linalg.norm(corner - final_corner) < minDistance:
                    too_close = True
                    break
            if not too_close:
                final_corners.append(corner)
        corners = np.array(final_corners)[:maxCorners].reshape(-1, 1, 2)
    else:
        corners = corners[:maxCorners].reshape(-1, 1, 2)

    return corners.astype(np.float32)


def corner_min_eigen_val(src, blockSize, ksize, borderType=cv2.BORDER_DEFAULT):
    """
    Рассчитывает минимальное собственное значение каждого "угла" в изображении.

    :param src: Исходное изображение.
    :param blockSize: Размер блока для локального углового детектора.
    :param ksize: Размер апертуры для оператора Собеля.
    :param borderType: Тип граничного условия.
    :return: Изображение минимальных собственных значений углов.
    """
    # Проверяем, что исходное изображение не пустое
    if src is None or src.size == 0:
        raise ValueError("Пустое исходное изображение")

    # Создаем матрицу для результатов с тем же размером и типом, что и исходное изображение
    dst = np.empty(src.shape, dtype=np.float32)

    # Рассчитываем минимальные собственные значения для каждого блока изображения
    dst = cv2.cornerMinEigenVal(src, blockSize, ksize, borderType=borderType)

    return dst


def corner_eigen_vals_vecs(src, block_size, aperture_size, op_type, k=0., borderType=cv2.BORDER_DEFAULT):
    depth = src.dtype
    scale = (1 << ((aperture_size if aperture_size > 0 else 3) - 1)) * block_size
    if aperture_size < 0:
        scale *= 2.
    if depth == np.uint8:
        scale *= 255.
    scale = 1. / scale

    assert src.ndim == 2 and (depth == np.uint8 or depth == np.float32), "src must be 2D and either CV_8UC1 or CV_32FC1"

    # Вычисляем градиенты с использованием операторов Собеля или Шарра
    if aperture_size > 0:
        Dx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=aperture_size, scale=scale, delta=0, borderType=borderType)
        Dy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=aperture_size, scale=scale, delta=0, borderType=borderType)
    else:
        Dx = cv2.Scharr(src, cv2.CV_32F, 1, 0, scale=scale, delta=0, borderType=borderType)
        Dy = cv2.Scharr(src, cv2.CV_32F, 0, 1, scale=scale, delta=0, borderType=borderType)

    # Рассчитываем элементы ковариационной матрицы градиентов
    cov_data = np.empty(src.shape + (3,), dtype=np.float32)
    cov_data[..., 0] = Dx ** 2
    cov_data[..., 1] = Dx * Dy
    cov_data[..., 2] = Dy ** 2

    # Применяем box-фильтр для локального усреднения
    cov_data = cv2.boxFilter(cov_data, -1, (block_size, block_size), borderType=borderType)

    eigenv = np.empty_like(cov_data)
    if op_type == "MINEIGENVAL":
        # Рассчитываем минимальное собственное значение
        # Примечание: Для демонстрации, реализация может отличаться
        eigenv = cv2.cornerMinEigenVal(src, blockSize=block_size)
    elif op_type == "HARRIS":
        # Рассчитываем угловой отклик Харриса
        # Примечание: Для демонстрации, реализация может отличаться
        eigenv = cv2.cornerHarris(src, blockSize=block_size, ksize=aperture_size, k=k)
    elif op_type == "EIGENVALSVECS":
        # Рассчитываем собственные значения и векторы
        # Этот вариант требует дополнительной реализации
        pass

    return eigenv