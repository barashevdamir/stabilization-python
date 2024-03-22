import numpy as np
import cv2
from optflowlk import compute_optical_flow
from sklearn.cluster import KMeans

def save_video(output_path, frames, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

def lucas_kanade_optical_flow(prev_frame, current_frame, p0):

    p1, status = compute_optical_flow(prev_frame, current_frame, p0)
    # Фильтруем точки по статусу и преобразуем p0 к тому же формату, что и p1

    good_new = p1[status == 1].reshape(-1, 2)
    good_old = p0[status == 1].reshape(-1, 1, 2).reshape(-1, 2)
    if len(good_new) > 0 and len(good_old) > 0:
        # Вычисляем смещения для каждой оси
        shift_x = good_new[:, 0] - good_old[:, 0]
        shift_y = good_new[:, 1] - good_old[:, 1]

        # Вычисляем среднее смещение
        mean_shift_x = np.mean(shift_x)
        mean_shift_y = np.mean(shift_y)
    else:
        mean_shift_x, mean_shift_y = 0, 0

    return int(mean_shift_x), int(mean_shift_y), p1, status

def lucas_kanade_optical_flow_max_shift(prev_frame, current_frame, p0):

    p1, status = compute_optical_flow(prev_frame, current_frame, p0)
    # Фильтруем точки по статусу и преобразуем p0 к тому же формату, что и p1

    good_new = p1[status == 1].reshape(-1, 2)
    good_old = p0[status == 1].reshape(-1, 1, 2).reshape(-1, 2)
    if len(good_new) > 0 and len(good_old) > 0:
        # Вычисляем смещения для каждой оси
        shift_x = good_new[:, 0] - good_old[:, 0]
        shift_y = good_new[:, 1] - good_old[:, 1]

        # Вычисляем максимальное смещение
        max_shift_x = np.max(np.abs(shift_x))
        max_shift_y = np.max(np.abs(shift_y))

        # Определяем направление максимального смещения
        max_shift_x *= np.sign(np.mean(shift_x))
        max_shift_y *= np.sign(np.mean(shift_y))
    else:
        max_shift_x, max_shift_y = 0, 0

    return int(max_shift_x), int(max_shift_y), p1, status


def lucas_kanade_optical_flow_cluster_shifts(prev_frame, current_frame, p0, n_clusters=2):
    # Вычисляем оптический поток
    p1, status = compute_optical_flow(prev_frame, current_frame, p0)
    good_new = p1[status == 1].reshape(-1, 2)
    good_old = p0[status == 1].reshape(-1, 1, 2).reshape(-1, 2)

    # Вычисляем смещения
    shifts = good_new - good_old

    # Применяем K-средние к смещениям
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(shifts)

    # Находим кластер с наибольшим числом точек
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = labels[np.argmax(counts)]

    # Вычисляем среднее смещение в доминирующем кластере
    dominant_shift = np.mean(shifts[kmeans.labels_ == dominant_cluster], axis=0)

    # Разделяем dominant_shift на смещения по осям X и Y
    dominant_shift_x, dominant_shift_y = dominant_shift

    return dominant_shift_x, dominant_shift_y, p1, status

def draw_tracks(frame, p0, p1, status):
    """
    Рисует стрелки для отслеживаемых точек.
    """
    for i, (new, old) in enumerate(zip(p1, p0)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        if status[i]:
            # Рисуем стрелку от старой точки к новой
            frame = cv2.arrowedLine(frame, (c, d), (a, b), (0, 255, 0), 2, tipLength=0.3)
    return frame


def stabilize_video_with_custom_optical_flow(video_path):

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception("Unable to open video file!")

    # Получение информации о видео
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, prev_frame = video_capture.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(
        prev_frame_gray,
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
        useHarrisDetector=True
    )

    stabilized_frames = [prev_frame]

    while True:
        ret, current_frame = video_capture.read()
        if not ret:
            break

        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # shift_x, shift_y, p1, status = lucas_kanade_optical_flow(prev_frame_gray, current_frame_gray, p0)
        # shift_x, shift_y, p1, status = lucas_kanade_optical_flow_max_shift(prev_frame_gray, current_frame_gray, p0)
        shift_x, shift_y, p1, status = lucas_kanade_optical_flow_cluster_shifts(prev_frame_gray, current_frame_gray, p0)

        # Применяем смещение к текущему кадру
        M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
        stabilized_frame = cv2.warpAffine(current_frame, M, (frame_width, frame_height))

        stabilized_frames.append(stabilized_frame)

        # Визуализируем треки на стабилизированном кадре
        stabilized_frame_with_tracks = draw_tracks(stabilized_frame.copy(), p0, p1, status.flatten())

        cv2.imshow("Stabilized Frame with Tracks", stabilized_frame_with_tracks)

        # Обновляем prev_frame_gray и p0 для следующего кадра
        prev_frame_gray = current_frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(prev_frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7, useHarrisDetector=True)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    # Сохраняем стабилизированное видео
    output_video_path = "output_stabilized_video.mp4"
    save_video(output_video_path, stabilized_frames, fps, (frame_width, frame_height))

    video_capture.release()

# Пример использования
video_path = "data/TEST2.MP4"
stabilize_video_with_custom_optical_flow(video_path)

