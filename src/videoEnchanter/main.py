import logging
from services.VideoProcessor import VideoProcessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    video_path = input("Add meg a videó elérési útját: ").strip()

    processor = VideoProcessor()
    processor.process(video_path)


if __name__ == "__main__":
    main()