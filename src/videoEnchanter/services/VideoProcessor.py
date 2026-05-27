import logging
import queue
import threading
import time
import cv2

from services.VideoReader import VideoReader
from services.VideoWriter import VideoWriter
from services.FrameEnhancer import FrameEnhancer
from utils.FileUtils import buildOutputPath


logger = logging.getLogger(__name__)

SENTINEL = object()


class VideoProcessor:

    def __init__(
        self,
        queue_size: int = 8,
        enhancement_profile: str = "fast"
    ):
        self.queue_size = queue_size
        self.video_reader = VideoReader()
        self.video_writer = VideoWriter()
        self.frame_enhancer = FrameEnhancer(enhancement_profile)

    def _read_frames(self, capture, read_queue, state):
        try:
            while True:
                start_time = time.perf_counter()
                success, frame = capture.read()
                state["read_time"] += time.perf_counter() - start_time

                if not success:
                    read_queue.put(SENTINEL)
                    return

                read_queue.put(frame)
        except Exception as error:
            state["reader_error"] = error
            read_queue.put(SENTINEL)

    def _write_frames(self, writer, write_queue, state):
        try:
            while True:
                frame = write_queue.get()

                if frame is SENTINEL:
                    return

                start_time = time.perf_counter()
                writer.write(frame)
                state["write_time"] += time.perf_counter() - start_time
        except Exception as error:
            state["writer_error"] = error

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

        read_queue = queue.Queue(maxsize=self.queue_size)
        write_queue = queue.Queue(maxsize=self.queue_size)
        state = {
            "read_time": 0.0,
            "enhance_time": 0.0,
            "write_time": 0.0,
            "reader_error": None,
            "writer_error": None
        }

        reader_thread = threading.Thread(
            target=self._read_frames,
            args=(capture, read_queue, state),
            daemon=True
        )
        writer_thread = threading.Thread(
            target=self._write_frames,
            args=(writer, write_queue, state),
            daemon=True
        )

        total_start_time = time.perf_counter()
        reader_thread.start()
        writer_thread.start()

        current_frame = 0
        progress_interval = 250 if frame_count > 0 else 1
        progress_log_interval_seconds = 5.0
        last_progress_log_time = total_start_time

        while True:
            frame = read_queue.get()

            if frame is SENTINEL:
                break

            if state["reader_error"] is not None:
                raise state["reader_error"]

            enhance_start_time = time.perf_counter()
            enhanced_frame = self.frame_enhancer.enhance(frame)
            state["enhance_time"] += time.perf_counter() - enhance_start_time

            write_queue.put(enhanced_frame)

            current_frame += 1
            now = time.perf_counter()

            if (
                current_frame == 1
                or current_frame == frame_count
                or current_frame % progress_interval == 0
                or now - last_progress_log_time >= progress_log_interval_seconds
            ):
                elapsed_time = now - total_start_time
                frames_per_second = (
                    current_frame / elapsed_time
                    if elapsed_time > 0 else 0.0
                )
                logger.info(
                    "Feldolgozott frame: %s/%s (%.2f%%, %.2f frame/s)",
                    current_frame,
                    frame_count,
                    (current_frame / frame_count) * 100 if frame_count > 0 else 0.0,
                    frames_per_second
                )
                last_progress_log_time = now

        write_queue.put(SENTINEL)
        reader_thread.join()
        writer_thread.join()

        capture.release()
        writer.release()

        if state["reader_error"] is not None:
            raise state["reader_error"]

        if state["writer_error"] is not None:
            raise state["writer_error"]

        elapsed_time = time.perf_counter() - total_start_time
        average_frame_ms = (
            (elapsed_time / current_frame) * 1000
            if current_frame > 0 else 0.0
        )

        logger.info(
            "Videó feldolgozás kész. frame=%s teljes=%.2fs atlag=%.2fms olvasas=%.2fs javitas=%.2fs iras=%.2fs",
            current_frame,
            elapsed_time,
            average_frame_ms,
            state["read_time"],
            state["enhance_time"],
            state["write_time"]
        )