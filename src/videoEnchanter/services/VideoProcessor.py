import logging
import cv2

from services.VideoReader import VideoReader
from services.VideoWriter import VideoWriter
from services.FrameEnhancer import FrameEnhancer
from utils.FileUtils import buildOutputPath


logger = logging.getLogger(__name__)


class VideoProcessor:

    def __init__(self):
        self.video_reader = VideoReader()
        self.video_writer = VideoWriter()
        self.frame_enhancer = FrameEnhancer()

    def process(self, video_path: str):
        logger.info("Videó megnyitása...")

        capture = self.video_reader.open(video_path)

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = buildOutputPath(video_path)

        logger.info(f"Output videó: {output_path}")

        writer = self.video_writer.create(
            output_path,
            fps,
            width,
            height
        )

        current_frame = 0

        while True:
            success, frame = capture.read()

            if not success:
                break

            enhanced_frame = self.frame_enhancer.enhance(frame)

            writer.write(enhanced_frame)

            current_frame += 1

            logger.info(
                f"Feldolgozott frame: {current_frame}/{frame_count}"
            )

        capture.release()
        writer.release()

        logger.info("Videó feldolgozás kész.")