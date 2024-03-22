import cv2
import numpy as np
import math
from ctypes import POINTER, c_float

def find_transform_ecc(template_image, input_image, warp_matrix, motion_type, input_mask=None, gauss_filt_size=0):
    """
    Находит оптимальное пространственное преобразование между шаблонным и входным изображениями.

    Алгоритм использует метод Enhanced Correlation Coefficient (ECC) для максимизации корреляции
    между преобразованным входным изображением и шаблоном.

    Параметры:
    - template_image: np.ndarray, шаблонное изображение.
    - input_image: np.ndarray, входное изображение, которое будет преобразовано для совмещения с шаблоном.
    - warp_matrix: np.ndarray, начальное приближение матрицы преобразования.
    - motion_type: int, тип преобразования (например, cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN).
    - input_mask: np.ndarray (опционально), маска, определяющая область входного изображения, которая будет использоваться.
    - gauss_filt_size: int (опционально), размер ядра Гауссова фильтра для предварительного сглаживания изображений.

    Примечание:
    Этот метод итеративно оптимизирует параметры преобразования, используя градиентные методы, до схождения алгоритма
    или достижения заданного числа итераций. Результат может зависеть от качества начального приближения и параметров алгоритма.

    Возвращает:
    - Обновленная матрица преобразования, оптимизированная для максимизации корреляции между шаблоном и преобразованным входным изображением.
    - Значение корреляции между шаблоном и преобразованным изображением после последней итерации.

    """

    assert template_image.size != 0 and input_image.size != 0, "One of the images is empty"
    assert template_image.shape == input_image.shape, "Image shapes do not match"
    assert template_image.dtype == input_image.dtype, "Image data types do not match"

    if motion_type == cv2.MOTION_HOMOGRAPHY:
        assert warp_matrix.shape[1] == 3

    if warp_matrix.size == 0:
        row_count = 3 if motion_type == cv2.MOTION_HOMOGRAPHY else 2
        warp_matrix = np.eye(row_count, 3, dtype=np.float32)

    param_temp = 6  # default: affine
    if motion_type == cv2.MOTION_TRANSLATION:
        param_temp = 2
    elif motion_type == cv2.MOTION_EUCLIDEAN:
        param_temp = 3
    elif motion_type == cv2.MOTION_HOMOGRAPHY:
        param_temp = 8

    # Подготовка изображений
    if template_image.ndim == 2:
        template_image = template_image[:, :, np.newaxis]
    if input_image.ndim == 2:
        input_image = input_image[:, :, np.newaxis]

    # Подготовка изображений
    if input_mask is not None:
        _, pre_mask = cv2.threshold(input_mask, 0, 1, cv2.THRESH_BINARY)
    else:
        pre_mask = np.ones(input_image.shape, dtype=np.uint8)

    # Применение Гауссова фильтра, если это указано
    if gauss_filt_size > 0:
        template_image = cv2.GaussianBlur(template_image, (gauss_filt_size, gauss_filt_size), 0)
        input_image = cv2.GaussianBlur(input_image, (gauss_filt_size, gauss_filt_size), 0)

    template_image = template_image.astype(np.float32)
    input_image = input_image.astype(np.float32)

    ws, hs = template_image.shape[1], template_image.shape[0]
    templateZM = np.zeros((hs, ws), dtype=np.float32)

    # Создание сетки координат
    x, y = np.meshgrid(np.arange(ws), np.arange(hs))

    grad_x = cv2.Sobel(input_image, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(input_image, cv2.CV_32F, 0, 1, ksize=5)

    # Основной цикл оптимизации
    # Объявление переменных для итерационного процесса
    # Установка значений по умолчанию для критериев остановки
    number_of_iterations = 200
    termination_eps = 1e-10

    rho = -1
    last_rho = -termination_eps
    best_rho = rho
    best_warp_matrix = warp_matrix


    for iter in range(number_of_iterations):
        # Применение текущего преобразования к изображению и маске
        if motion_type != cv2.MOTION_HOMOGRAPHY:
            warped_image = cv2.warpAffine(input_image, warp_matrix[:2], (ws, hs), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            warped_grad_x = cv2.warpAffine(grad_x, warp_matrix[:2], (ws, hs),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            warped_grad_y = cv2.warpAffine(grad_y, warp_matrix[:2], (ws, hs),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            warped_image = cv2.warpPerspective(input_image, warp_matrix, (ws, hs), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            warped_grad_x = cv2.warpPerspective(grad_x, warp_matrix, (ws, hs),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            warped_grad_y = cv2.warpPerspective(grad_y, warp_matrix, (ws, hs),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        imgMean, imgStd = cv2.meanStdDev(warped_image, pre_mask)
        tmpMean, tmpStd = cv2.meanStdDev(template_image, pre_mask)
        warped_image -= imgMean
        templateZM -= tmpMean
        tmpNorm = np.sqrt(np.count_nonzero(pre_mask) * tmpStd * tmpStd)
        imgNorm = np.sqrt(np.count_nonzero(pre_mask) * imgStd * imgStd)

        # Определение ядер для градиентов
        dx = np.array([-0.5, 0.0, 0.5]).reshape(1, 3)
        dy = dx.T

        # Применение cv2.filter2D для вычисления градиентов
        grad_x = cv2.filter2D(warped_image, -1, dx)
        grad_y = cv2.filter2D(warped_image, -1, dy)

        # Расчет Якобиана в зависимости от типа движения
        if motion_type == cv2.MOTION_AFFINE:
            hessian_size = 6
            jacobian_size = ws * 6  # Размер для Афинного преобразования
            return
        elif motion_type == cv2.MOTION_HOMOGRAPHY:
            hessian_size = 8
            jacobian_size = ws * 8  # Размер для Гомографии
            return
        elif motion_type == cv2.MOTION_TRANSLATION:
            hessian_size = 2
            jacobian_size = ws * 2  # Размер для Трансляции
            return
        elif motion_type == cv2.MOTION_EUCLIDEAN:
            hessian_size = 3
            jacobian_size = ws * 3  # Размер для Евклидова преобразования
            jacobian = compute_euclidean_jacobian(x, y, warped_image)


        # Проецирование на Якобиан
        # Инициализация hessian
        hessian = project_onto_jacobian(jacobian, jacobian, use_for='hessian')

        hessianInv = np.linalg.inv(hessian)
        correlation = np.dot(templateZM.flatten(), warped_image.flatten())

        last_rho = rho
        rho = correlation / (imgNorm * tmpNorm)

        imageProjection = project_onto_jacobian(jacobian, warped_image.flatten(), use_for='error')
        templateProjection = project_onto_jacobian(jacobian, templateZM.flatten(), use_for='error')
        imageProjectionHessian = np.dot(hessianInv, imageProjection)
        lambda_n = imgNorm * imgNorm - np.dot(imageProjection.flatten(), imageProjectionHessian.flatten())
        lambda_d = correlation - np.dot(templateProjection.flatten(), imageProjectionHessian.flatten())
        lambda_ = lambda_n / lambda_d
        error = lambda_ * templateZM - warped_image
        errorProjection = project_onto_jacobian(jacobian, error.flatten(), use_for='error')
        delta_p = np.dot(hessianInv, errorProjection)

        # Обновление матрицы преобразования
        # warp_matrix = update_warping_matrix_ECC(warp_matrix, delta_p, motion_type)
        warp_matrix = update_warping_matrix(warp_matrix, delta_p, motion_type)

        print(f"Iteration {iter + 1}, Correlation: {rho}, abs(new_rho - rho): {abs(last_rho - rho)}")
        if abs(last_rho - rho) < termination_eps:
        # if abs(1 - new_rho) < termination_eps:
            # Обновление rho
            best_rho = rho
            best_warp_matrix = warp_matrix.copy()
            print("Достигнута высокая степень корреляции. Останавливаем итерации.")
            break

    # Возвращаем обновленную матрицу преобразования и последнюю корреляцию
    return best_warp_matrix, best_rho


def calculate_pearson_correlation(image1, image2):
    """
    Вычисляет коэффициент корреляции Пирсона между двумя изображениями.

    Параметры:
    - image1, image2: np.ndarray, два изображения в градациях серого.

    Возвращает:
    - Коэффициент корреляции Пирсона.
    """
    if image1.shape != image2.shape:
        raise ValueError("Изображения должны иметь одинаковый размер.")

    image1_flatten = image1.flatten()
    image2_flatten = image2.flatten()

    mean1 = np.mean(image1_flatten)
    mean2 = np.mean(image2_flatten)

    covariance = np.mean((image1_flatten - mean1) * (image2_flatten - mean2))
    std1 = np.std(image1_flatten)
    std2 = np.std(image2_flatten)

    if std1 > 0 and std2 > 0:
        correlation = covariance / (std1 * std2)
    else:
        correlation = 0  # Нет вариации одного из изображений

    return correlation

def compute_euclidean_jacobian(x, y, image):

    # Вычисление градиентов изображения
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=5)

    # Вычисляем компоненты Якобиана для евклидовых преобразований
    jacobian_theta = -y * grad_x + x * grad_y
    jacobian_tx = grad_x
    jacobian_ty = grad_y

    # Формируем Якобиан
    jacobian = np.stack([jacobian_theta, jacobian_tx, jacobian_ty], axis=-1)

    return jacobian

def project_onto_jacobian(jacobian, target, use_for='hessian'):
    """
    Проецирует целевую матрицу или вектор (target) на Якобиан (jacobian).

    Параметры:
    - jacobian: np.ndarray, матрица Якобиана изображения.
    - target: np.ndarray, матрица или вектор для проецирования на Якобиан.
               Для Гессиана target будет jacobian, для проекции ошибки — разность изображений.
    - use_for: str, определяет, для чего используется проекция ('hessian' или 'error').

    Возвращает:
    - Результат проецирования: Гессиан или проекцию ошибки.
    """
    h, w, params = jacobian.shape
    if use_for == 'hessian':
        projection = np.zeros((params, params))
        for i in range(params):
            for j in range(i, params):
                proj_temp = jacobian[:, :, i] * jacobian[:, :, j]
                projection[i, j] = proj_temp.sum()
                if i != j:
                    projection[j, i] = projection[i, j]  # Гессиан симметричен
    elif use_for == 'error':
        # Преобразуем target к требуемой форме, если он одномерный
        if target.ndim == 1:
            target = target.reshape((h, w, 1))
        projection = np.zeros(params)
        for i in range(params):
            proj_temp = jacobian[:, :, i] * target.squeeze()
            projection[i] = proj_temp.sum()
    else:
        raise ValueError("Неизвестное использование проекции")

    return projection


def update_warping_matrix(warp_matrix, delta_p, motion_type):
    if motion_type == cv2.MOTION_TRANSLATION:
        # Для трансляции delta_p содержит [dx, dy]
        assert delta_p.shape[0] == 2, "Ожидалось 2 параметра обновления для трансляции."
        dx, dy = delta_p.flatten()
        warp_matrix[0, 2] += dx
        warp_matrix[1, 2] += dy

    elif motion_type == cv2.MOTION_EUCLIDEAN:
        # Для евклидова преобразования delta_p содержит [dtheta, dx, dy]
        assert delta_p.shape[0] == 3, "Ожидалось 3 параметра обновления для евклидова преобразования."
        dtheta, dx, dy = delta_p.flatten()

        # Вычисляем новый угол поворота и обновляем матрицу
        theta = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0]) + dtheta
        new_warp_matrix = np.array([
            [np.cos(theta), -np.sin(theta), dx],
            [np.sin(theta), np.cos(theta), dy],
            [0, 0, 1]
        ], dtype=np.float32)
        return new_warp_matrix[:2]

    elif motion_type == cv2.MOTION_AFFINE:
        # Для аффинного преобразования delta_p содержит [da0, da1, da2, da3, da4, da5]
        assert delta_p.shape[0] == 6, "Ожидалось 6 параметров обновления для аффинного преобразования."
        warp_matrix += delta_p.reshape(2, 3)

    return warp_matrix
