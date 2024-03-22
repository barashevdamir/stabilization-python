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

    # Создание сетки координат
    x, y = np.meshgrid(np.arange(ws), np.arange(hs))


    # Основной цикл оптимизации
    # Объявление переменных для итерационного процесса
    # Установка значений по умолчанию для критериев остановки
    number_of_iterations = 200
    termination_eps = 1e-10

    rho = -1
    best_rho = rho
    best_warp_matrix = warp_matrix


    for iter in range(number_of_iterations):
        # Применение текущего преобразования к изображению и маске
        if motion_type != cv2.MOTION_HOMOGRAPHY:
            warped_image = cv2.warpAffine(input_image, warp_matrix[:2], (ws, hs), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            # warped_grad_x = cv2.warpAffine(grad_x, warp_matrix[:2], (ws, hs),
            #                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            # warped_grad_y = cv2.warpAffine(grad_y, warp_matrix[:2], (ws, hs),
            #                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            warped_image = cv2.warpPerspective(input_image, warp_matrix, (ws, hs), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            # warped_grad_x = cv2.warpPerspective(grad_x, warp_matrix, (ws, hs),
            #                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            # warped_grad_y = cv2.warpPerspective(grad_y, warp_matrix, (ws, hs),
            #                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

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
            jacobian = np.zeros((hs, jacobian_size), dtype=np.float32)
            jacobian = image_jacobian_affine_ECC(grad_x, grad_y, x, y, jacobian)
        elif motion_type == cv2.MOTION_HOMOGRAPHY:
            hessian_size = 8
            jacobian_size = ws * 8  # Размер для Гомографии
            jacobian = np.zeros((hs, jacobian_size), dtype=np.float32)
            jacobian = image_jacobian_homo_ECC(grad_x, grad_y, x, y, warp_matrix, jacobian)
        elif motion_type == cv2.MOTION_TRANSLATION:
            hessian_size = 2
            jacobian_size = ws * 2  # Размер для Трансляции
            jacobian = np.zeros((hs, jacobian_size), dtype=np.float32)
            jacobian = image_jacobian_translation_ECC(grad_x, grad_y, jacobian)
        elif motion_type == cv2.MOTION_EUCLIDEAN:
            hessian_size = 3
            jacobian_size = ws * 3  # Размер для Евклидова преобразования
            jacobian = np.zeros((hs, jacobian_size), dtype=np.float32)
            # jacobian = image_jacobian_euclidean_ECC(grad_x, grad_y, x, y, warp_matrix, jacobian)
            jacobian = compute_euclidean_jacobian(x, y, warped_image)


        # Проецирование на Якобиан
        # Инициализация hessian
        hessian = np.zeros((hessian_size, hessian_size), dtype=np.float32)
        # hessian = project_onto_jacobian_ECC(jacobian, jacobian, hessian)
        hessian = project_onto_jacobian(jacobian, jacobian, use_for='hessian')

        # Преобразование warped_image к форме (1080,1920,1) перед вычитанием
        # Преобразование разности изображений к подходящему формату
        error_img = template_image - warped_image[:, :, np.newaxis]
        # error_projection = np.zeros((hessian_size, hessian_size), dtype=np.float32)
        # error_projection = project_onto_jacobian_ECC(jacobian, error_img, error_projection)
        error_projection = project_onto_jacobian(jacobian, error_img, use_for='error')

        # Расчет обновления для матрицы преобразования
        delta_p = np.linalg.inv(hessian) @ error_projection
        delta_p = delta_p.reshape(-1, 1)  # Преобразование в двумерный массив с одним столбцом

        # Обновление матрицы преобразования
        # warp_matrix = update_warping_matrix_ECC(warp_matrix, delta_p, motion_type)
        warp_matrix = update_warping_matrix(warp_matrix, delta_p, motion_type)

        # Подсчёт корреляции (можно использовать другие метрики)
        # new_rho = np.corrcoef(template_image.flatten(), warped_image.flatten())[0, 1]
        new_rho = calculate_pearson_correlation(template_image.flatten(), warped_image.flatten())

        print(f"Iteration {iter + 1}, Correlation: {new_rho}, abs(new_rho - rho): {abs(new_rho - rho)}")
        if abs(new_rho - rho) < termination_eps:
        # if abs(1 - new_rho) < termination_eps:
            # Обновление rho
            best_rho = new_rho
            best_warp_matrix = warp_matrix.copy()
            print("Достигнута высокая степень корреляции. Останавливаем итерации.")
            break

        rho = new_rho

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

def image_jacobian_homo_ECC(src1, src2, src3, src4, src5, dst):
    """
    Вычисляет Якобиан изображения для гомографии.

    Функция использует 5 исходных изображений (или производные от них величины) и преобразованное изображение,
    для вычисления Якобиана гомографии между исходным и преобразованным изображением. Якобиан вычисляется
    с использованием коэффициентов гомографии, представленных вектором `src5`.

    Параметры:
        src1, src2, src3, src4: np.ndarray
            Исходные изображения или соответствующие матрицы, участвующие в вычислении Якобиана.
            Все эти массивы должны иметь одинаковую форму.
        src5: np.ndarray
            Вектор, содержащий коэффициенты гомографии. Должен быть одномерным и поддерживать порядок данных 'C'.
        dst: np.ndarray
            Матрица назначения для хранения вычисленного Якобиана. Её форма должна соответствовать форме
            `[src1.shape[0], src1.shape[1] * 8]`, а тип данных - np.float32.

    Примечание:
        - Проверяется соответствие форм и типов входных данных, чтобы обеспечить корректность вычислений.
        - Вычисление включает в себя создание знаменателя для проекций точек, вычисление самих проекций,
          предварительное деление блоков градиентов и, наконец, вычисление блоков Якобиана.

    Возвращает:
        np.ndarray: Матрица `dst`, содержащая вычисленный Якобиан.

    """
    assert src1.shape == src2.shape == src3.shape == src4.shape
    assert src1.shape[0] == dst.shape[0]
    assert dst.shape[1] == src1.shape[1] * 8
    assert dst.dtype == np.float32
    assert src5.flags['C_CONTIGUOUS']

    h = src5.ravel()

    h0_, h1_, h2_ = h[0], h[3], h[6]
    h3_, h4_, h5_ = h[1], h[4], h[7]
    h6_, h7_ = h[2], h[5]

    w = src1.shape[1]

    epsilon = 1e-10

    # Создание знаменателя для всех точек
    den_ = src3 * h2_ + src4 * h5_ + 1.0 + epsilon

    # Создание проекций точек
    hatX_ = -(src3 * h0_ + src4 * h3_ + h6_)
    hatX_ /= den_
    hatY_ = -(src3 * h1_ + src4 * h4_ + h7_)
    hatY_ /= den_

    # Предварительное деление блоков градиентов
    src1Divided_ = src1 / den_
    src2Divided_ = src2 / den_

    # Вычисление блоков Якобиана (8 блоков)
    dst[:, 0:w] = src1Divided_ * src3
    dst[:, w:2*w] = src2Divided_ * src3
    temp_ = (hatX_ * src1Divided_ + hatY_ * src2Divided_) * src3
    dst[:, 2*w:3*w] = temp_

    dst[:, 3*w:4*w] = src1Divided_ * src4
    dst[:, 4*w:5*w] = src2Divided_ * src4
    dst[:, 5*w:6*w] = temp_ * src4

    dst[:, 6*w:7*w] = src1Divided_
    dst[:, 7*w:8*w] = src2Divided_

    return dst

# def image_jacobian_euclidean_ECC(src1, src2, src3, src4, src5, dst):
#     """
#     Вычисляет Якобиан изображения для евклидовых преобразований.
#
#     Функция строит Якобиан для преобразований изображений, основанных на евклидовых
#     преобразованиях (вращения и трансляции). Это включает в себя вычисление производных
#     изображения по углу вращения и по параметрам трансляции.
#
#     Параметры:
#         src1 (np.ndarray): Градиент изображения по оси X.
#         src2 (np.ndarray): Градиент изображения по оси Y.
#         src3 (np.ndarray): Сетка координат X изображения.
#         src4 (np.ndarray): Сетка координат Y изображения.
#         src5 (np.ndarray): Матрица преобразования (предполагается единичной матрицей с дополнительными параметрами).
#         dst (np.ndarray): Выходная матрица Якоби.
#
#     Примечание:
#         - src1, src2, src3, и src4 должны иметь одинаковый размер.
#         - Высота src1 должна совпадать с высотой dst.
#         - Ширина dst должна быть в три раза больше ширины src1.
#         - dst должен быть типа np.float32.
#         - src5 должен быть C-последовательным.
#
#     Возвращает:
#         np.ndarray: Заполненная матрица Якоби для евклидовых преобразований.
#
#     """
#     assert src1.shape == src2.shape == src3.shape == src4.shape
#     assert src1.shape[0] == dst.shape[0]
#     assert dst.shape[1] == src1.shape[1] * 3
#     assert dst.dtype == np.float32
#     assert src5.flags['C_CONTIGUOUS']
#
#     h = src5.ravel()
#
#     h0 = h[0]  # cos(theta)
#     h1 = h[3]  # sin(theta)
#
#     w = src1.shape[1]
#
#     # Создание -sin(theta)*X - cos(theta)*Y для всех точек
#     hatX = -(src3 * h1) - (src4 * h0)
#
#     # Создание cos(theta)*X - sin(theta)*Y для всех точек
#     hatY = (src3 * h0) - (src4 * h1)
#
#     # Вычисление блоков Якобиана (3 блока)
#     dst[:, 0:w] = (src1 * hatX) + (src2 * hatY)  # 1
#
#     dst[:, w:2*w] = src1.copy()  # 2
#     dst[:, 2*w:3*w] = src2.copy()  # 3
#
#     return dst

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

def image_jacobian_affine_ECC(src1, src2, src3, src4, dst):
    """
    Вычисляет Якобиан изображения для аффинных преобразований.

    Эта функция рассчитывает производные изображения по параметрам аффинного
    преобразования, которое включает в себя масштабирование, вращение, сдвиг и трансляцию.
    Результат заполняется в матрицу Якоби. Для каждой точки изображения строится блок Якоби,
    содержащий производные по шести параметрам аффинного преобразования.

    Параметры:
        src1 (np.ndarray): Матрица градиента изображения по оси X.
        src2 (np.ndarray): Матрица градиента изображения по оси Y.
        src3 (np.ndarray): Сетка координат X изображения.
        src4 (np.ndarray): Сетка координат Y изображения.
        dst (np.ndarray): Выходная матрица Якоби, которая будет заполнена.

    Примечание:
        - src1, src2, src3 и src4 должны иметь одинаковые размеры.
        - Высота src1 должна совпадать с высотой dst.
        - Ширина dst должна быть в шесть раз больше ширины src1.
        - dst должен быть типа np.float32.

    Возвращает:
        np.ndarray: Заполненная матрица Якоби для аффинного преобразования.
    """
    assert src1.shape == src2.shape == src3.shape == src4.shape
    assert src1.shape[0] == dst.shape[0]
    assert dst.shape[1] == 6 * src1.shape[1]
    assert dst.dtype == np.float32

    w = src1.shape[1]

    # Вычисление блоков Якобиана (6 блоков)
    dst[:, 0:w] = src1 * src3  # 1
    dst[:, w:2*w] = src2 * src3  # 2
    dst[:, 2*w:3*w] = src1 * src4  # 3
    dst[:, 3*w:4*w] = src2 * src4  # 4
    dst[:, 4*w:5*w] = src1  # 5
    dst[:, 5*w:6*w] = src2  # 6

    return dst

def image_jacobian_translation_ECC(src1, src2, dst):
    """
    Вычисляет Якобиан изображения относительно параметров трансляции.

    Данная функция предполагает, что преобразование изображения моделируется
    как чистая трансляция (сдвиг) без масштабирования или вращения. Она заполняет
    матрицу Якоби блоками, каждый из которых представляет собой частные производные
    изображения по параметрам трансляции.

    Параметры:
        src1 (np.ndarray): Матрица градиента изображения по оси X.
        src2 (np.ndarray): Матрица градиента изображения по оси Y.
        dst (np.ndarray): Выходная матрица Якоби, которая будет заполнена.
                          Предполагается, что она уже инициализирована
                          соответствующими размерами и типом данных.

    Примечание:
        - src1 и src2 должны иметь одинаковые размеры.
        - Ширина dst должна быть удвоенной ширины src1 и src2.
        - Высота src1 должна совпадать с высотой dst.
        - dst должен быть типа np.float32.

    Возвращает:
        np.ndarray: Заполненная матрица Якоби для трансляции.
    """
    assert src1.shape == src2.shape
    assert src1.shape[0] == dst.shape[0]
    assert dst.shape[1] == src1.shape[1] * 2
    assert dst.dtype == np.float32

    w = src1.shape[1]

    # Вычисление блоков Якобиана (2 блока)
    dst[:, 0:w] = src1  # 1
    dst[:, w:2*w] = src2  # 2

    return dst

# def project_onto_jacobian_ECC(src1, src2, dst):
#     """
#     Проектирует одну матрицу на другую, используя блочное умножение, для вычисления вклада
#     каждого параметра преобразования в изменение изображения. Это используется в процессе
#     оптимизации ECC для обновления параметров преобразования.
#
#     Если число столбцов src1 не равно числу столбцов src2, dst будет вектором размера
#     (number_of_blocks x 1), где каждый элемент является результатом скалярного произведения
#     соответствующего блока src1 на src2. Это происходит, когда src2 представляет собой вектор
#     градиентов изображения, а src1 - Якобиан преобразования.
#
#     Если число столбцов src1 равно числу столбцов src2, dst будет матрицей размера
#     (number_of_blocks x number_of_blocks), где каждый элемент представляет блочное умножение
#     блоков матриц src1 и src2. Это используется для вычисления гессиана функции стоимости.
#
#     Параметры:
#         src1 : np.ndarray
#             Первая входная матрица.
#         src2 : np.ndarray
#             Вторая входная матрица.
#         dst : np.ndarray
#             Результирующая матрица, куда будет записан результат.
#
#     Возвращает:
#         np.ndarray: Заполненная матрица Якоби для трансляции.
#     """
#     assert src1.shape[0] == src2.shape[0], "src1 и src2 должны иметь одинаковое количество строк"
#     assert (src1.shape[1] % src2.shape[1]) == 0, "Количество столбцов src1 должно быть кратно количеству столбцов src2"
#
#     if src1.shape[1] != src2.shape[1]:
#         # В этом случае предполагается, что src2 - это вектор
#         w = src2.shape[1]
#         for i in range(dst.shape[0]):
#             # Важно: src1 должен быть подготовлен так, чтобы соответствовать размерам src2 для умножения
#             # Здесь может потребоваться изменить логику для соответствия вашим математическим намерениям
#             dst[i] = np.dot(src1[:, i*w:(i+1)*w].reshape(-1), src2.reshape(-1))
#     else:
#         # Когда src1 и src2 имеют одинаковое количество столбцов
#         w = src2.shape[1] // dst.shape[1]
#         for i in range(dst.shape[0]):
#             mat = src1[:, i*w:(i+1)*w]
#             dst[i, i] = np.linalg.norm(mat)**2  # Вычисление диагональных элементов
#
#             for j in range(i+1, dst.shape[1]):  # Заполнение остальных элементов матрицы
#                 dst[i, j] = np.dot(mat.flatten(), src2[:, j*w:(j+1)*w].flatten())
#                 dst[j, i] = dst[i, j]  # Использование симметрии матрицы
#
#     return dst

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
        # Для Гессиана target является самим Якобианом
        projection = np.zeros((params, params))
        for i in range(params):
            for j in range(i, params):
                proj_temp = jacobian[:, :, i] * jacobian[:, :, j]
                projection[i, j] = proj_temp.sum()
                if i != j:
                    projection[j, i] = projection[i, j]  # Гессиан симметричен
    elif use_for == 'error':
        # Для проекции ошибки target — это разность изображений
        assert target.ndim == 3, "Target должен иметь 3 измерения"
        projection = np.zeros(params)
        for i in range(params):
            proj_temp = jacobian[:, :, i] * target.squeeze()
            projection[i] = proj_temp.sum()
    else:
        raise ValueError("Неизвестное использование проекции")

    return projection


# def update_warping_matrix_ECC(map_matrix, update, motionType):
#     """
#     Обновляет матрицу преобразования для алгоритма ECC, применяя к ней вектор обновления.
#     Поддерживает различные типы движения: трансляцию, евклидово преобразование, аффинное
#     преобразование и гомографию.
#
#     Параметры:
#     map_matrix : np.ndarray
#         Текущая матрица преобразования, которая будет обновлена.
#     update : np.ndarray
#         Вектор обновления, содержащий изменения параметров преобразования.
#     motionType : int
#         Тип движения, который определяет, как применять обновление.
#         Может быть одним из следующих: MOTION_TRANSLATION, MOTION_EUCLIDEAN,
#         MOTION_AFFINE, MOTION_HOMOGRAPHY.
#     """
#     assert map_matrix.dtype == np.float32
#     assert update.dtype == np.float32
#     assert motionType in [cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, cv2.MOTION_HOMOGRAPHY]
#     assert update.shape[1] == 1
#
#     if motionType == cv2.MOTION_TRANSLATION:
#         map_matrix[0, 2] += update[0]
#         map_matrix[1, 2] += update[1]
#     elif motionType == cv2.MOTION_AFFINE:
#         map_matrix[0, 0] += update[0]
#         map_matrix[1, 0] += update[1]
#         map_matrix[0, 1] += update[2]
#         map_matrix[1, 1] += update[3]
#         map_matrix[0, 2] += update[4]
#         map_matrix[1, 2] += update[5]
#     elif motionType == cv2.MOTION_HOMOGRAPHY:
#         assert map_matrix.shape[0] == 3
#         map_matrix[0, :] += update[:3].flatten()
#         map_matrix[1, :] += update[3:6].flatten()
#         map_matrix[2, :2] += update[6:8].flatten()
#     elif motionType == cv2.MOTION_EUCLIDEAN:
#         theta_update = update[0]
#         new_theta = np.arcsin(map_matrix[1, 0]) + theta_update
#         cos_theta = np.cos(new_theta)
#         sin_theta = np.sin(new_theta)
#
#         map_matrix[0, 0] = map_matrix[1, 1] = cos_theta
#         map_matrix[1, 0] = sin_theta
#         map_matrix[0, 1] = -sin_theta
#         map_matrix[0, 2] += update[1]
#         map_matrix[1, 2] += update[2]
#
#     return map_matrix

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
