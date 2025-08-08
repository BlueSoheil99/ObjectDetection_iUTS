import cv2
import os


def trimVideo(input_path, output_path, trim_minutes):
    '''
    Trims video and saves trimmed video.
    :param input_path:
    :param output_path:
    :param trim_minutes:
    :return:
    '''

    # === Open the video ===
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    print(f"Video FPS: {fps}")
    print(f"Video duration: {duration_sec / 60:.2f} minutes")

    # === Compute frame limit ===
    trim_seconds = trim_minutes * 60
    max_frames = int(min(total_frames, fps * trim_seconds))

    # === Define VideoWriter ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    # === Cleanup ===
    cap.release()
    out.release()
    print(f"Trimmed video saved to: {output_path}")


if __name__ == '__main__':
    IN = 'data/input/sdot15NE40_20240322_0800_to_0810.avi'
    OUT = 'data/input/trimmed_vid.mp4'
    MIN = 2
    trimVideo(IN, OUT, MIN)
