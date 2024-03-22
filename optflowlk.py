import numpy as np
from scipy.signal import convolve2d
import cv2
import time
from Sobel import sobel

def gaussian_kernel(size, sigma=1.0):
    """Создание Гауссова ядра."""
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def apply_sobel_filters_open_cv(img):
    """Применение фильтров Собеля для вычисления градиентов Ix и Iy."""
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return Ix, Iy

def apply_sobel_filters(img):
    """Применение фильтров Собеля с генерацией ядер для вычисления градиентов Ix и Iy."""
    # Получение ядер Собеля

    # Применение свёртки с полученными ядрами для вычисления градиентов
    Ix = sobel(img, dx=1, dy=0, ksize=3)
    Iy = sobel(img, dx=0, dy=1, ksize=3)
    return Ix, Iy


def compute_optical_flow(prev_img, next_img, p0, win_size=5, gauss_sigma=1.5):
    """Вычисление оптического потока."""
    prev_img = prev_img.astype(np.float32)
    next_img = next_img.astype(np.float32)


    Ix, Iy = apply_sobel_filters_open_cv(prev_img)
    # Ix, Iy = apply_sobel_filters(prev_img)
    It = next_img - prev_img

    G = gaussian_kernel(win_size, sigma=gauss_sigma)

    p1 = []
    status = []

    for point in p0:
        x, y = point.ravel()  # Изменение здесь
        x_min = max(int(x - win_size // 2), 0)
        y_min = max(int(y - win_size // 2), 0)
        x_max = min(int(x + win_size // 2 + 1), prev_img.shape[1])
        y_max = min(int(y + win_size // 2 + 1), prev_img.shape[0])

        Ix_win = Ix[y_min:y_max, x_min:x_max].flatten()
        Iy_win = Iy[y_min:y_max, x_min:x_max].flatten()
        It_win = It[y_min:y_max, x_min:x_max].flatten()

        G_win = G[int(y_min - y + win_size // 2):int(y_max - y + win_size // 2),
                int(x_min - x + win_size // 2):int(x_max - x + win_size // 2)].flatten()

        A = np.vstack((Ix_win * G_win, Iy_win * G_win)).T
        b = -It_win * G_win

        # Решение системы уравнений
        nu, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # if residuals.size > 0 and residuals[0] < 1e-2:
        #
        #     p1.append([x + nu[0], y + nu[1]])
        #     status.append(1)
        # else:
        #     p1.append([x, y])
        #     status.append(0)
        # Проверяем, успешно ли было решение
        if nu.size == 2:  # Просто проверяем, что решение имеет два компонента
            p1.append([x + nu[0], y + nu[1]])
            status.append(1)
        else:
            p1.append([x, y])
            status.append(0)

    return np.array(p1).reshape(-1, 1, 2), np.array(status)
