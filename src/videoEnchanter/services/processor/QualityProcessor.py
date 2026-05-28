import time
from concurrent.futures import ProcessPoolExecutor

from videoEnchanter.services.frameEnhancer.QualityFrameEnhancer import QualityFrameEnhancer
from videoEnchanter.services.processor.Processor import Processor, SENTINEL, logger


ENHANCER = None


def _init_enhancer_worker(profile):
    global ENHANCER
    if profile != "quality":
        raise ValueError(f"Nem támogatott quality enhancer profile: {profile}")

    ENHANCER = QualityFrameEnhancer()


def _enhance_frame_worker(frame_index, frame):
    start_time = time.perf_counter()
    enhanced_frame = ENHANCER.enhance(frame)
    return frame_index, enhanced_frame, time.perf_counter() - start_time


class QualityProcessor(Processor):

    @property
    def enhancement_profile(self):
        return "quality"

    def should_process(self, profile: str):
        return profile == self.enhancement_profile

    def _log_processing_mode(self):
        if self.worker_count > 1:
            logger.info("Enhancer mod: process-pool (%s worker)", self.worker_count)
            return

        logger.info("Enhancer mod: single-worker")

    def process(self, video_path: str):
        self._process_video(video_path)

    def _create_enhancer_pool(self):
        return ProcessPoolExecutor(
            max_workers=self.worker_count,
            initializer=_init_enhancer_worker,
            initargs=(self.enhancement_profile,)
        )

    def _submit_frame(self, executor, pending_futures, frame_index, frame):
        pending_futures[frame_index] = executor.submit(
            _enhance_frame_worker,
            frame_index,
            frame
        )

    def _drain_completed_frames(self, pending_futures, next_frame_to_write, write_queue, state):
        current_frame = 0

        while next_frame_to_write in pending_futures:
            future = pending_futures[next_frame_to_write]

            if not future.done():
                break

            _, enhanced_frame, enhance_time = future.result()
            state["enhance_time"] += enhance_time
            write_queue.put(enhanced_frame)
            del pending_futures[next_frame_to_write]
            next_frame_to_write += 1
            current_frame += 1

        return next_frame_to_write, current_frame

    def _flush_pending_frames(self, pending_futures, next_frame_to_write, write_queue, state):
        current_frame = 0

        while next_frame_to_write in pending_futures:
            _, enhanced_frame, enhance_time = pending_futures[next_frame_to_write].result()
            state["enhance_time"] += enhance_time
            write_queue.put(enhanced_frame)
            del pending_futures[next_frame_to_write]
            next_frame_to_write += 1
            current_frame += 1

        return current_frame

    def _process_frames_in_pool(self, read_queue, write_queue, state, frame_count, total_start_time):
        current_frame = 0
        next_frame_index = 0
        next_frame_to_write = 0
        last_progress_log_time = total_start_time
        pending_futures = {}
        max_pending_frames = max(self.queue_size, self.worker_count * 2)

        with self._create_enhancer_pool() as executor:
            while True:
                frame = read_queue.get()

                if frame is SENTINEL:
                    break

                self._validate_worker_state(state)
                self._submit_frame(executor, pending_futures, next_frame_index, frame)
                next_frame_index += 1

                if len(pending_futures) >= max_pending_frames:
                    drained_count = 0

                    while drained_count == 0:
                        next_frame_to_write, drained_count = self._drain_completed_frames(
                            pending_futures,
                            next_frame_to_write,
                            write_queue,
                            state
                        )

                    current_frame += drained_count
                    last_progress_log_time = self._log_pending_progress(
                        current_frame,
                        frame_count,
                        total_start_time,
                        last_progress_log_time
                    )

                next_frame_to_write, drained_count = self._drain_completed_frames(
                    pending_futures,
                    next_frame_to_write,
                    write_queue,
                    state
                )

                if drained_count > 0:
                    current_frame += drained_count
                    last_progress_log_time = self._log_pending_progress(
                        current_frame,
                        frame_count,
                        total_start_time,
                        last_progress_log_time
                    )

            current_frame += self._flush_pending_frames(
                pending_futures,
                next_frame_to_write,
                write_queue,
                state
            )

        self._log_pending_progress(
            current_frame,
            frame_count,
            total_start_time,
            last_progress_log_time
        )
        return current_frame

    def _process_frames(self, read_queue, write_queue, state, frame_count, total_start_time):
        if self.worker_count <= 1:
            from videoEnchanter.services.processor.FastProcessor import FastProcessor

            fallback_processor = FastProcessor(queue_size=self.queue_size, worker_count=self.worker_count)
            return fallback_processor._process_frames(
                read_queue,
                write_queue,
                state,
                frame_count,
                total_start_time
            )

        return self._process_frames_in_pool(
            read_queue,
            write_queue,
            state,
            frame_count,
            total_start_time
        )