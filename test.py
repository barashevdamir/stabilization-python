import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import convolve
import os


# from jupyterthemes import jtplot
# jtplot.style(theme='grade3', grid=False, ticks=True, context='paper', figsize=(20, 15), fscale=1.4)


def load_images(PATH, OUT_PATH=None):
    """
    Загружает изображения из видеофайла.

    Args:
        PATH (str): Путь к видеофайлу.
        OUT_PATH (str, optional): Путь для сохранения изображений. По умолчанию None.

    Returns:
        list: Список загруженных изображений.

    """
    cap = cv2.VideoCapture(PATH)
    again = True
    i = 0
    imgs = []
    while again:
        again, img = cap.read()
        if again:
            imgs += [img]
            if not OUT_PATH is None:
                os.makedirs(OUT_PATH, exist_ok=True)
                filename = OUT_PATH + "".join([str(0)] * (3 - len(str(i)))) + str(i) + '.png'
                cv2.imwrite(filename, img)
            i += 1
        else:
            break
    return imgs


def create_gif(filenames, PATH):
    """
    Создает GIF-анимацию из списка изображений и сохраняет ее.

    Args:
        filenames (list): Список путей к изображениям.
        PATH (str): Путь для сохранения GIF-анимации.

    """
    kwargs = {'duration': 0.0333}
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(PATH, images, **kwargs)


def imshow_with_trajectory(images, warp_stack, PATH, ij):
    """
    Визуализирует траекторию движения объекта на последовательности изображений.

    Args:
        images (list): Список изображений.
        warp_stack (ndarray): Массив траекторий движения объекта.
        PATH (str): Путь для сохранения изображений с визуализацией траектории.
        ij (tuple): Кортеж координат (i, j) для выбора соответствующей траектории.

    Returns:
        list: Список созданных файлов с изображениями с визуализацией траектории.

    """
    traj_dict = {(0, 0): 'Width', (0, 1): 'sin(Theta)', (1, 0): '-sin(Theta)', (1, 1): 'Height', (0, 2): 'X',
                 (1, 2): 'Y'}
    i, j = ij
    filenames = []
    for k in range(1, len(warp_stack)):
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

        a0.axis('off')
        a0.imshow(images[k])

        a1.plot(np.arange(len(warp_stack)), np.cumsum(warp_stack[:, i, j]))
        a1.scatter(k, np.cumsum(warp_stack[:, i, j])[k], c='r', s=100)
        a1.set_xlabel('Frame')
        a1.set_ylabel(traj_dict[ij] + ' Trajectory')

        if not PATH is None:
            os.makedirs(PATH, exist_ok=True)
            filename = PATH + "".join([str(0)] * (3 - len(str(k)))) + str(k) + '.png'
            plt.savefig(filename)
            filenames += [filename]
        plt.close()
    return filenames


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
    print('bottom', maxmin[:, 0].argmax() // 2)
    top = maxmin[:, 0].min()
    print('top', maxmin[:, 0].argmin() // 2)
    left = maxmin[:, 1].min()
    print('right', maxmin[:, 1].argmax() // 2)
    right = maxmin[:, 1].max()
    print('left', maxmin[:, 1].argmin() // 2)
    return int(-top), int(bottom - img_shape[0]), int(-left), int(right - img_shape[1])



def get_homography(img1, img2, motion=cv2.MOTION_EUCLIDEAN):
    """
    Определяет матрицу гомографии между двумя изображениями.

    Args:
        img1 (ndarray): Первое изображение.
        img2 (ndarray): Второе изображение.
        motion (int, optional): Тип движения, который нужно найти. По умолчанию `cv2.MOTION_EUCLIDEAN`.

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
    warp_matrix = cv2.findTransformECC(templateImage=imga, inputImage=imgb, warpMatrix=warpMatrix, motionType=motion)[1]
    return warp_matrix

def create_warp_stack(imgs):
    """
    Создает стек гомографий между последовательными кадрами видео.

    Args:
        imgs (list): Список изображений, представляющих видеокадры.

    Returns:
        ndarray: Массив, содержащий гомографии между каждой парой последовательных кадров видео.

    """
    warp_stack = []
    for i, img in enumerate(imgs[:-1]):
        warp_stack += [get_homography(img, imgs[i + 1])]
    return np.array(warp_stack)



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
            kernel = signal.gaussian(1000, sigma_mat[i, j])
            kernel = kernel / np.sum(kernel)
            smoothed_trajectory[:, i, j] = convolve(original_trajectory[:, i, j], kernel, mode='reflect')
    smoothed_warp = np.apply_along_axis(lambda m: convolve(m, [0, 1, -1], mode='reflect'), axis=0,
                                        arr=smoothed_trajectory)
    smoothed_warp[:, 0, 0] = 0
    smoothed_warp[:, 1, 1] = 0
    return smoothed_warp, smoothed_trajectory, original_trajectory


def apply_warping_fullview(images, warp_stack, PATH=None):
    """
    Применяет преобразование перспективы к изображениям с учетом траектории warp_stack.

    Args:
        images (List[ndarray]): Список изображений для преобразования.
        warp_stack (ndarray): Трехмерный массив, содержащий набор траекторий warp_stack.
        PATH (str, optional): Путь для сохранения преобразованных изображений. По умолчанию None.

    Returns:
        List[ndarray]: Список преобразованных изображений.

    """
    top, bottom, left, right = get_border_pads(img_shape=images[0].shape, warp_stack=warp_stack)
    H = homography_gen(warp_stack)
    imgs = []
    for i, img in enumerate(images[1:]):
        H_tot = next(H) + np.array([[0, 0, left], [0, 0, top], [0, 0, 0]])
        img_warp = cv2.warpPerspective(img, H_tot, (img.shape[1] + left + right, img.shape[0] + top + bottom))
        if not PATH is None:
            os.makedirs(PATH, exist_ok=True)
            filename = PATH + "".join([str(0)] * (3 - len(str(i)))) + str(i) + '.png'
            cv2.imwrite(filename, img_warp)
        imgs += [img_warp]
    return imgs


def plot_velocity_and_trajectory(ws, name):
    i, j = 0, 2
    plt.scatter(np.arange(len(ws)), ws[:, i, j], label='X Velocity')
    plt.plot(np.arange(len(ws)), ws[:, i, j])
    plt.scatter(np.arange(len(ws)), np.cumsum(ws[:, i, j], axis=0), label='X Trajectory')
    plt.plot(np.arange(len(ws)), np.cumsum(ws[:, i, j], axis=0))
    plt.legend()
    plt.xlabel('Frame')
    plt.savefig(name + '_trajectory.png')


def plot_original_and_smoothed_trajectory(original_trajectory, smoothed_trajectory, name):
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})

    i, j = 0, 2
    a0.scatter(np.arange(len(original_trajectory)), np.array(original_trajectory)[:, i, j], label='Original')
    a0.plot(np.arange(len(original_trajectory)), np.array(original_trajectory)[:, i, j])
    a0.scatter(np.arange(len(smoothed_trajectory)), np.array(smoothed_trajectory)[:, i, j], label='Smoothed')
    a0.plot(np.arange(len(smoothed_trajectory)), np.array(smoothed_trajectory)[:, i, j])
    a0.legend()
    a0.set_ylabel('X trajectory')
    a0.xaxis.set_ticklabels([])

    i, j = 0, 1
    a1.scatter(np.arange(len(original_trajectory)), np.array(original_trajectory)[:, i, j], label='Original')
    a1.plot(np.arange(len(original_trajectory)), np.array(original_trajectory)[:, i, j])
    a1.scatter(np.arange(len(smoothed_trajectory)), np.array(smoothed_trajectory)[:, i, j], label='Smoothed')
    a1.plot(np.arange(len(smoothed_trajectory)), np.array(smoothed_trajectory)[:, i, j])
    a1.legend()
    a1.set_xlabel('Frame')
    a1.set_ylabel('Sin(Theta) trajectory')
    plt.savefig(name + '_smoothed.png')


imgs, name = load_images('data/TEST4.mp4', OUT_PATH='./frames_result1/'), 'result1'
ws = create_warp_stack(imgs)
# Построение графиков скорости и траектории
plot_velocity_and_trajectory(ws, name)

#calculate the smoothed trajectory and output the zeroed images
smoothed_warp, smoothed_trajectory, original_trajectory = moving_average(ws, sigma_mat= np.array([[1000,15, 10],[15,1000, 10]]))
new_imgs = apply_warping_fullview(images=imgs, warp_stack=ws-smoothed_warp, PATH='./out/')

# Построение графиков исходной и сглаженной траектории
plot_original_and_smoothed_trajectory(original_trajectory, smoothed_trajectory, name)

#create a images that show both the trajectory and video frames
filenames = imshow_with_trajectory(images=new_imgs, warp_stack=ws-smoothed_warp, PATH='./out_'+name+'/', ij=(0,2))

# Создание GIF из изображений в папке out
image_filenames = [os.path.join('./out/', f"{i:03d}.png") for i in range(len(new_imgs))]
#create gif
create_gif(image_filenames, './'+name+'.gif')

