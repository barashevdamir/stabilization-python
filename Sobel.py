import numpy as np


def get_sobel_kernels(dx, dy, _ksize, normalize=False, ktype=np.float32):
    assert ktype in [np.float32, np.float64], "ktype must be np.float32 or np.float64"

    if _ksize == 1 and dx > 0:
        ksizeX = 3
    else:
        ksizeX = _ksize

    if _ksize == 1 and dy > 0:
        ksizeY = 3
    else:
        ksizeY = _ksize

    assert _ksize % 2 == 1 and _ksize <= 31, "The kernel size must be odd and not larger than 31"
    assert dx >= 0 and dy >= 0 and dx + dy > 0, "dx and dy should be non-negative and at least one greater than 0"

    kx = np.zeros((ksizeX, 1), dtype=ktype)
    ky = np.zeros((ksizeY, 1), dtype=ktype)

    for k, kernel in enumerate([kx, ky]):
        order = dx if k == 0 else dy
        ksize = ksizeX if k == 0 else ksizeY
        kerI = [1] + [0] * (ksize - 1)

        for i in range(ksize - order - 1):
            oldval = kerI[0]
            for j in range(1, ksize + 1):
                newval = kerI[j] + kerI[j - 1] if j < ksize else 0
                kerI[j - 1] = oldval
                oldval = newval

        for i in range(order):
            oldval = -kerI[0]
            for j in range(1, ksize + 1):
                newval = kerI[j - 1] - kerI[j] if j < ksize else 0
                kerI[j - 1] = oldval
                oldval = newval

        scale = 1.0 if not normalize else 1.0 / (2 ** (ksize - order - 1))
        kernel[:, 0] = [x * scale for x in kerI[:ksize]]

    return kx, ky


def convolve(image, kernel):
    """Простая свёртка изображения с ядром."""
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # Добавление отступов к изображению
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Инициализация результата свёртки
    convolved = np.zeros_like(image)

    # Применение свёртки
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            convolved[i, j] = np.sum(region * kernel)

    return convolved


def sobel(img, dx=1, dy=0, ksize=3):
    """
    Применяет оператор Собеля к изображению для вычисления градиента по заданной оси.

    :param img: Исходное изображение.
    :param dx: Порядок производной по оси x.
    :param dy: Порядок производной по оси y.
    :param ksize: Размер ядра.
    :return: Результат применения оператора Собеля.
    """

    # Получение ядер Собеля
    kx, ky = get_sobel_kernels(dx, dy, ksize)

    # Применение ядер для вычисления градиентов
    if dx > 0:
        grad_x = convolve(img, kx)
    else:
        grad_x = np.zeros_like(img)

    if dy > 0:
        grad_y = convolve(img, ky)
    else:
        grad_y = np.zeros_like(img)

    # Выбор градиента в зависимости от заданных параметров
    if dx > 0 and dy == 0:
        return grad_x
    elif dy > 0 and dx == 0:
        return grad_y
    else:
        # В случае, если заданы оба направления, возвращаем комбинированный градиент
        return np.sqrt(grad_x ** 2 + grad_y ** 2)

#
# kx, ky = get_sobel_kernels(1, 0, 3, normalize=False, ktype=np.float32)
# print(f'My: {kx, ky}')
#
# import cv2
#
# kx, ky = cv2.getDerivKernels(1, 0, 3, normalize=False, ktype=cv2.CV_32F)
# print(f'OpenCV: {kx, ky}')
