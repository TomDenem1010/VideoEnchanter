from videoEnchanter.services.processor.FastProcessor import FastProcessor
from videoEnchanter.services.processor.QualityProcessor import QualityProcessor


class VideoProcessor:

    def __init__(
        self,
        queue_size: int = 8,
        enhancement_profile: str = "fast",
        worker_count: int | None = None
    ):
        self.enhancement_profile = enhancement_profile
        self.processors = [
            FastProcessor(queue_size=queue_size, worker_count=worker_count),
            QualityProcessor(queue_size=queue_size, worker_count=worker_count)
        ]

    def process(self, video_path: str):
        for processor in self.processors:
            if processor.should_process(self.enhancement_profile):
                processor.process(video_path)
                return

        raise ValueError(f"Nem támogatott enhancement profile: {self.enhancement_profile}")