import logging
import os
import queue
import shutil
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2

from videoEnchanter.services.file.AudioMuxer import AudioMuxer
from videoEnchanter.services.file.VideoReader import VideoReader
from videoEnchanter.services.file.VideoWriter import VideoWriter
from videoEnchanter.services.file.FileUtils import buildOutputPath, buildTemporaryOutputPath


logger = logging.getLogger(__name__)

SENTINEL = object()


class Processor(ABC):

    def __init__(self, queue_size: int = 8, worker_count: int | None = None):
        self.queue_size = queue_size
        self.worker_count = worker_count or self._get_default_worker_count()
        self.audio_muxer = AudioMuxer()
        self.video_reader = VideoReader()
        self.video_writer = VideoWriter()

    def _get_default_worker_count(self):
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count - 1)

    @property
    @abstractmethod
    def enhancement_profile(self):
        pass

    @abstractmethod
    def should_process(self, profile: str):
        pass

    @abstractmethod
    def _log_processing_mode(self):
        pass

    @abstractmethod
    def _process_frames(self, read_queue, write_queue, state, frame_count, total_start_time):
        pass

    @abstractmethod
    def process(self, video_path: str):
        pass

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

    def _read_video_metadata(self, capture):
        return {
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        }

    def _create_processing_state(self):
        return {
            "read_time": 0.0,
            "enhance_time": 0.0,
            "write_time": 0.0,
            "reader_error": None,
            "writer_error": None
        }

    def _create_queues(self):
        return (
            queue.Queue(maxsize=self.queue_size),
            queue.Queue(maxsize=self.queue_size)
        )

    def _start_worker_threads(self, capture, writer, read_queue, write_queue, state):
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

        reader_thread.start()
        writer_thread.start()

        return reader_thread, writer_thread

    def _should_log_progress(self, current_frame, frame_count, now, last_log_time):
        return (
            current_frame == frame_count
            or now - last_log_time >= 5.0
        )

    def _log_progress(self, current_frame, frame_count, total_start_time, now):
        elapsed_time = now - total_start_time
        frames_per_second = current_frame / elapsed_time if elapsed_time > 0 else 0.0
        progress_percent = (
            (current_frame / frame_count) * 100
            if frame_count > 0 else 0.0
        )
        remaining_frames = max(frame_count - current_frame, 0)
        remaining_minutes = (
            (remaining_frames / frames_per_second) / 60
            if frame_count > 0 and frames_per_second > 0 else 0.0
        )
        logger.info(
            "Feldolgozott frame: %s/%s (%.2f%%, %.2f frame/s, hatra: %.1f perc)",
            current_frame,
            frame_count,
            progress_percent,
            frames_per_second,
            remaining_minutes
        )

    def _validate_worker_state(self, state):
        if state["reader_error"] is not None:
            raise state["reader_error"]

        if state["writer_error"] is not None:
            raise state["writer_error"]

    def _finalize_processing(self, capture, writer, write_queue, reader_thread, writer_thread):
        write_queue.put(SENTINEL)
        reader_thread.join()
        writer_thread.join()
        capture.release()
        writer.release()

    def _log_pending_progress(self, current_frame, frame_count, total_start_time, last_progress_log_time):
        now = time.perf_counter()

        if self._should_log_progress(current_frame, frame_count, now, last_progress_log_time):
            self._log_progress(current_frame, frame_count, total_start_time, now)
            return now

        return last_progress_log_time

    def _log_summary(self, total_start_time, current_frame, state):
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

    def _finalize_output(self, source_video_path: str, temporary_output_path: str, output_path: str):
        temporary_file = Path(temporary_output_path)

        if not temporary_file.exists():
            raise FileNotFoundError(f"Nem található az ideiglenes kimeneti videó: {temporary_output_path}")

        if Path(output_path).exists():
            Path(output_path).unlink()

        try:
            self.audio_muxer.merge_audio(
                source_video_path,
                temporary_output_path,
                output_path
            )
        finally:
            if temporary_file.exists() and temporary_file.resolve() != Path(output_path).resolve():
                temporary_file.unlink()

    def _process_video(self, video_path: str):
        logger.info("Videó megnyitása...")
        self._log_processing_mode()

        capture = self.video_reader.open(video_path)
        metadata = self._read_video_metadata(capture)

        output_path = buildOutputPath(video_path, self.enhancement_profile)
        temporary_output_path = buildTemporaryOutputPath(output_path)

        logger.info(f"Output videó: {output_path}")

        writer = self.video_writer.create(
            temporary_output_path,
            metadata["fps"],
            metadata["width"],
            metadata["height"]
        )

        read_queue, write_queue = self._create_queues()
        state = self._create_processing_state()

        total_start_time = time.perf_counter()
        reader_thread, writer_thread = self._start_worker_threads(
            capture,
            writer,
            read_queue,
            write_queue,
            state
        )

        frame_count = metadata["frame_count"]
        current_frame = self._process_frames(
            read_queue,
            write_queue,
            state,
            frame_count,
            total_start_time
        )

        self._finalize_processing(
            capture,
            writer,
            write_queue,
            reader_thread,
            writer_thread
        )
        self._validate_worker_state(state)
        self._finalize_output(video_path, temporary_output_path, output_path)
        self._log_summary(total_start_time, current_frame, state)