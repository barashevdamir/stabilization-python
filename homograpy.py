import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve
import os
from findTransformECC import find_transform_ecc
from transform import find_transform_ecc as transform


def get_homography(img1, img2, motion=cv2.MOTION_EUCLIDEAN):
    """
    Определяет матрицу гомографии между двумя изображениями.

    Args:
        img1 (ndarray): Первое изображение.
        img2 (ndarray): Второе изображение.
        motion (int, optional): Тип движения, который определяет, как применять обновление.
                Может быть одним из следующих: MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE, MOTION_HOMOGRAPHY.

    Returns:
        ndarray: Матрица гомографии между двумя изображениями.

    """
    imga = img1.copy().astype(np.float32)
    imgb = img2.copy().astype(np.float32)
    if len(imga.shape) == 3:
        imga = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
    if len(imgb.shape) == 3:
        imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    if motion == cv2.MOTION_HOMOGRAPHY:
        warpMatrix = np.eye(3, 3, dtype=np.float32)
    else:
        warpMatrix = np.eye(2, 3, dtype=np.float32)

    # warp_matrix, rho = find_transform_ecc(template_image=imga, input_image=imgb, warp_matrix=warpMatrix, motion_type=motion)
    # warp_matrix, rho = transform(template_image=imga, input_image=imgb, warp_matrix=warpMatrix, motion_type=motion)

    warp_matrix = cv2.findTransformECC(templateImage=imga, inputImage=imgb, warpMatrix=warpMatrix, motionType=motion)[1]
    # print(cv2.findTransformECC(templateImage=imga, inputImage=imgb, warpMatrix=warpMatrix, motionType=motion)[0])
    return warp_matrix

def get_border_pads(img_shape, warp_stack):
    """
    Определяет размеры дополнительных границ для изображений после применения трансформаций.

    Args:
        img_shape (tuple): Размеры изображений в формате (высота, ширина).
        warp_stack (ndarray): Массив трансформаций изображений.

    Returns:
        tuple: Кортеж значений (top, bottom, left, right), представляющий размеры верхней, нижней, левой и правой границ соответственно.

    """
    maxmin = []
    corners = np.array([[0, 0, 1], [img_shape[1], 0, 1], [0, img_shape[0], 1], [img_shape[1], img_shape[0], 1]]).T
    warp_prev = np.eye(3)
    for warp in warp_stack:
        warp = np.concatenate([warp, [[0, 0, 1]]])
        warp = np.matmul(warp, warp_prev)
        warp_invs = np.linalg.inv(warp)
        new_corners = np.matmul(warp_invs, corners)
        xmax, xmin = new_corners[0].max(), new_corners[0].min()
        ymax, ymin = new_corners[1].max(), new_corners[1].min()
        maxmin += [[ymax, xmax], [ymin, xmin]]
        warp_prev = warp.copy()
    maxmin = np.array(maxmin)
    bottom = maxmin[:, 0].max()
    # print('bottom', maxmin[:, 0].argmax() // 2)
    top = maxmin[:, 0].min()
    # print('top', maxmin[:, 0].argmin() // 2)
    left = maxmin[:, 1].min()
    # print('right', maxmin[:, 1].argmax() // 2)
    right = maxmin[:, 1].max()
    # print('left', maxmin[:, 1].argmin() // 2)
    return int(-top), int(bottom - img_shape[0]), int(-left), int(right - img_shape[1])

def homography_gen(warp_stack):
    """
    Генератор, который выдает обратные гомографии относительно начального положения кадров.

    Args:
        warp_stack (ndarray): Массив гомографий между последовательными кадрами видео.

    Yields:
        ndarray: Обратная гомография, представляющая преобразование от текущего кадра к начальному.

    """
    H_tot = np.eye(3)
    wsp = np.dstack([warp_stack[:, 0, :], warp_stack[:, 1, :], np.array([[0, 0, 1]] * warp_stack.shape[0])])
    for i in range(len(warp_stack)):
        H_tot = np.matmul(wsp[i].T, H_tot)
        yield np.linalg.inv(H_tot)  # [:2]

def create_warp_stack_from_video(cap):
    """
    Создает стек гомографий между последовательными кадрами видео.

    Args:
        cap: Объект cv2.VideoCapture

    Returns:
        ndarray: Массив, содержащий гомографии между каждой парой последовательных кадров видео.

    """
    imgs = []
    success, frame = cap.read()
    while success:
        imgs.append(frame)
        success, frame = cap.read()
    cap.release()

    warp_stack = []
    for i, img in enumerate(imgs[:-1]):
        warp_stack.append(get_homography(img, imgs[i + 1]))
    # Построение графика скорости и траектории камеры
    return np.array(warp_stack)

def moving_average(warp_stack, sigma_mat):
    """
    Применяет скользящее среднее к траектории warp_stack.

    Args:
        warp_stack (ndarray): Трехмерный массив, содержащий набор траекторий warp_stack.
        sigma_mat (ndarray): Матрица стандартных отклонений для каждой точки в warp_stack.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: Кортеж, содержащий:
            - Сглаженную траекторию warp_stack.
            - Сглаженную траекторию без изменений.
            - Исходную траекторию warp_stack.

    """
    x, y = warp_stack.shape[1:]
    original_trajectory = np.cumsum(warp_stack, axis=0)
    smoothed_trajectory = np.zeros(original_trajectory.shape)
    for i in range(x):
        for j in range(y):
            kernel = gaussian(1000, sigma_mat[i, j])
            kernel = kernel / np.sum(kernel)
            smoothed_trajectory[:, i, j] = convolve(original_trajectory[:, i, j], kernel, mode='reflect')
    smoothed_warp = np.apply_along_axis(lambda m: convolve(m, [0, 1, -1], mode='reflect'), axis=0,
                                        arr=smoothed_trajectory)
    smoothed_warp[:, 0, 0] = 0
    smoothed_warp[:, 1, 1] = 0
    return smoothed_warp, smoothed_trajectory, original_trajectory

def create_video_from_frames(frames, output_path, fps=30):
    """
    Создает видео из кадров и сохраняет его.

    Args:
        frames (List[ndarray]): Список изображений кадров.
        output_path (str): Путь для сохранения видеофайла.
        fps (int, optional): Количество кадров в секунду. По умолчанию 30.

    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

def apply_warping_fullview_to_video(video_path, warp_stack):
    """
    Применяет преобразование перспективы к каждому кадру видео с учетом траектории warp_stack.

    Args:
        video_path (str): Путь к видеофайлу.
        warp_stack (ndarray): Трехмерный массив, содержащий набор траекторий warp_stack.

    Returns:
        List[ndarray]: Список преобразованных кадров видео.

    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    H = homography_gen(warp_stack)
    print(warp_stack)
    transformed_frames = []

    success, img = cap.read()
    if not success:
        cap.release()
        return transformed_frames

    success, img = cap.read()
    while success:
        H_tot = next(H)
        img_warp = cv2.warpPerspective(img, H_tot, frame_size)
        transformed_frames.append(img_warp)
        success, img = cap.read()

    cap.release()
    return transformed_frames

def crop_frames_to_common_area(transformed_frames, warp_stack, original_aspect_ratio):
    """
    Обрезает преобразованные кадры до общей области с сохранением исходного соотношения сторон.

    Args:
        transformed_frames (List[ndarray]): Список кадров видео после преобразования.
        warp_stack (ndarray): Массив трансформаций для каждого кадра.
        original_aspect_ratio (float): Исходное соотношение сторон видео (ширина / высота).

    Returns:
        List[ndarray]: Список обрезанных кадров видео.
    """
    # Подсчет минимальных границ, включающих все кадры
    min_top, min_bottom, min_left, min_right = float('inf'), 0, float('inf'), 0
    for frame in transformed_frames:
        top, bottom, left, right = get_border_pads(frame.shape[:2], warp_stack)
        min_top = min(min_top, top)
        min_bottom = max(min_bottom, bottom)
        min_left = min(min_left, left)
        min_right = max(min_right, right)

    # Вычисляем новые размеры кадра с учетом соотношения сторон
    cropped_height = frame.shape[0] - min_top - min_bottom
    cropped_width = frame.shape[1] - min_left - min_right
    new_cropped_width = int(cropped_height * original_aspect_ratio)

    # Обрезаем кадры до результирующих границ, корректируя ширину для сохранения соотношения сторон
    if new_cropped_width <= cropped_width:
        # Уменьшаем ширину, если вычисленная ширина меньше текущей обрезанной ширины
        width_diff = cropped_width - new_cropped_width
        left_increase = width_diff // 2
        right_decrease = width_diff - left_increase
        min_left += left_increase
        min_right += right_decrease
    else:
        # Если новая ширина больше текущей обрезанной, корректируем высоту для сохранения соотношения сторон
        new_cropped_height = int(cropped_width / original_aspect_ratio)
        height_diff = cropped_height - new_cropped_height
        top_increase = height_diff // 2
        bottom_decrease = height_diff - top_increase
        min_top += top_increase
        min_bottom += bottom_decrease

    cropped_frames = []
    for frame in transformed_frames:
        cropped_frame = frame[min_top:frame.shape[0]-min_bottom, min_left:frame.shape[1]-min_right]
        cropped_frames.append(cropped_frame)

    return cropped_frames

def plot_original_and_smoothed_trajectory(original_trajectory, smoothed_trajectory, name):
    """
    Визуализирует траекторию и скорость камеры на основе исходной и сглаженной траекторий.

    Эта функция строит два графика: один для визуализации траекторий движения камеры (исходной и сглаженной),
    и второй для визуализации скорости камеры, вычисленной как производная траектории по времени. Траектория
    отображается в виде графика перемещения камеры по оси X и Y, а скорость — как изменение этих позиций
    между последовательными кадрами.

    Параметры:
        original_trajectory (ndarray): Массив Numpy, содержащий исходную траекторию камеры. Это должен быть
                                       трехмерный массив, где каждый элемент [i, :, :] представляет собой
                                       трансформационную матрицу для i-го кадра.
        smoothed_trajectory (ndarray): Массив Numpy, содержащий сглаженную траекторию камеры. Формат аналогичен
                                       original_trajectory.
        name (str): Путь для сохранения графика.


    Возвращает:
        None: Функция только сохраняет графики и не возвращает никаких значений.
    """
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})

    i, j = 0, 2
    a0.scatter(np.arange(len(original_trajectory)), np.array(original_trajectory)[:, i, j], label='Исходный')
    a0.plot(np.arange(len(original_trajectory)), np.array(original_trajectory)[:, i, j])
    a0.scatter(np.arange(len(smoothed_trajectory)), np.array(smoothed_trajectory)[:, i, j], label='Сглаженный')
    a0.plot(np.arange(len(smoothed_trajectory)), np.array(smoothed_trajectory)[:, i, j])
    a0.legend()
    a0.set_ylabel('X траектория')
    a0.xaxis.set_ticklabels([])

    i, j = 0, 1
    a1.scatter(np.arange(len(original_trajectory)), np.array(original_trajectory)[:, i, j], label='Исходный')
    a1.plot(np.arange(len(original_trajectory)), np.array(original_trajectory)[:, i, j])
    a1.scatter(np.arange(len(smoothed_trajectory)), np.array(smoothed_trajectory)[:, i, j], label='Сглаженный')
    a1.plot(np.arange(len(smoothed_trajectory)), np.array(smoothed_trajectory)[:, i, j])
    a1.legend()
    a1.set_xlabel('Кадр')
    a1.set_ylabel('Sin(Theta) траектория')
    plt.savefig(name + '_smoothed.png')

def plot_camera_motion(original_trajectory, smoothed_trajectory, title_prefix="", save_path=None):
    """
    Визуализирует траекторию и скорость камеры на основе исходной и сглаженной траекторий.

    Эта функция строит два графика: один для визуализации траекторий движения камеры (исходной и сглаженной),
    и второй для визуализации скорости камеры, вычисленной как производная траектории по времени. Траектория
    отображается в виде графика перемещения камеры по оси X и Y, а скорость — как изменение этих позиций
    между последовательными кадрами.

    Параметры:
        original_trajectory (ndarray): Массив Numpy, содержащий исходную траекторию камеры. Это должен быть
                                       трехмерный массив, где каждый элемент [i, :, :] представляет собой
                                       трансформационную матрицу для i-го кадра.
        smoothed_trajectory (ndarray): Массив Numpy, содержащий сглаженную траекторию камеры. Формат аналогичен
                                       original_trajectory.
        title_prefix (str, optional): Префикс для заголовков графиков. Позволяет добавить дополнительное описание
                                      к заголовкам графиков, например, для указания, что графики относятся к
                                      конкретному виду траектории или камере. По умолчанию пустая строка.
        save_path (str, optional): Путь для сохранения графика. Если параметр не указан, график будет отображён
                                   на экране без сохранения в файл.


    Возвращает:
        None: Функция только отображает и/или сохраняет графики и не возвращает никаких значений.
    """
    # Извлекаем координаты X и Y из траекторий
    original_x = original_trajectory[:, 0, 2]
    original_y = original_trajectory[:, 1, 2]
    smoothed_x = smoothed_trajectory[:, 0, 2]
    smoothed_y = smoothed_trajectory[:, 1, 2]

    # Вычисляем скорость как производную позиции по времени
    velocity_original = np.sqrt(np.diff(original_x) ** 2 + np.diff(original_y) ** 2)
    velocity_smoothed = np.sqrt(np.diff(smoothed_x) ** 2 + np.diff(smoothed_y) ** 2)

    # Строим график траектории
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(original_x, original_y, label="Исходная")
    plt.plot(smoothed_x, smoothed_y, label="Сглаженная")
    plt.title(f"{title_prefix}: Траектория")
    plt.legend()

    # Строим график скорости
    plt.subplot(1, 2, 2)
    plt.plot(velocity_original, label="Исходная скорость")
    plt.plot(velocity_smoothed, label="Сглаженная скорость")
    plt.title(f"{title_prefix}: Скорость")
    plt.legend()

    plt.tight_layout()
    # Проверяем, требуется ли сохранение графика в файл
    if save_path:
        # Создаем директории пути, если они не существуют
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()  # Закрываем фигуру, чтобы освободить ресурсы
    else:
        plt.show()  # Показываем график на экране, если путь для сохранения не указан

def plot_velocity_and_trajectory(warp_stack, name):
    i, j = 0, 2
    plt.scatter(np.arange(len(warp_stack)), warp_stack[:, i, j], label='X Скорость')
    plt.plot(np.arange(len(warp_stack)), warp_stack[:, i, j])
    plt.scatter(np.arange(len(warp_stack)), np.cumsum(warp_stack[:, i, j], axis=0), label='X Траектория')
    plt.plot(np.arange(len(warp_stack)), np.cumsum(warp_stack[:, i, j], axis=0))
    plt.legend()
    plt.xlabel('Кадр')
    plt.savefig(name + '_trajectory.png')

def stabilize_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception("Unable to open video file!")

    # Получение информации о видео
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_aspect_ratio = frame_width/frame_height

    warp_stack = create_warp_stack_from_video(video_capture)

    plot_velocity_and_trajectory(warp_stack, name="camera_motion")

    smoothed_warp, smoothed_trajectory, original_trajectory = moving_average(warp_stack, sigma_mat=np.array(
        [[1000, 15, 10], [15, 1000, 10]]))

    # Визуализируем траекторию и скорость камеры на основе исходной и сглаженной траектории
    plot_camera_motion(original_trajectory, smoothed_trajectory, title_prefix="Траектория камеры")
    plot_original_and_smoothed_trajectory(original_trajectory, smoothed_trajectory, name="camera_motion")

    transformed_frames = apply_warping_fullview_to_video(video_path, warp_stack - smoothed_warp)

    # Создание и сохранение видео
    output_video_path = 'output_video.mp4'
    create_video_from_frames(transformed_frames, output_video_path, fps=fps)
    # # После применения трансформаций ко всем кадрам:
    # cropped_transformed_frames = crop_frames_to_common_area(transformed_frames, warp_stack, original_aspect_ratio=original_aspect_ratio)
    #
    # create_video_from_frames(cropped_transformed_frames, output_video_path, fps=fps)


video_path = 'data/C00011.MP4'
print("Начинаю стабилизацию видео")
stabilize_video(video_path)
print("Стабилизация видео завершена")