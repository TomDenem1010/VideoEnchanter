import time

from videoEnchanter.constants.Type import FAST
from videoEnchanter.services.frameEnhancer.FastFrameEnhancer import FastFrameEnhancer
from videoEnchanter.services.processor.Processor import Processor, SENTINEL, logger


class FastProcessor(Processor):

    @property
    def enhancement_profile(self):
        return FAST

    def should_process(self, profile: str):
        return profile == self.enhancement_profile

    def _log_processing_mode(self):
        logger.info("Enhancer mod: single-worker")

    def process(self, video_path: str):
        self._process_video(video_path)

    def _process_frames(self, read_queue, write_queue, state, frame_count, total_start_time):
        current_frame = 0
        last_progress_log_time = total_start_time
        enhancer = FastFrameEnhancer()

        while True:
            frame = read_queue.get()

            if frame is SENTINEL:
                break

            self._validate_worker_state(state)
            start_time = time.perf_counter()
            enhanced_frame = enhancer.enhance(frame)
            state["enhance_time"] += time.perf_counter() - start_time
            write_queue.put(enhanced_frame)
            current_frame += 1
            last_progress_log_time = self._log_pending_progress(
                current_frame,
                frame_count,
                total_start_time,
                last_progress_log_time
            )

        return current_frame