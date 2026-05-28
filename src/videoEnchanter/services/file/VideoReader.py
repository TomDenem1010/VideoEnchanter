import logging
import cv2


logger = logging.getLogger(__name__)


class VideoReader:

    def open(self, video_path: str):
        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            logger.error(f"Nem sikerült megnyitni a videót: {video_path}")
            raise RuntimeError(f"Nem sikerült megnyitni a videót: {video_path}")

        return capture