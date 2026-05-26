import logging
from services.VideoProcessor import VideoProcessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

PROFILE_CHOICES = {
    "1": "fast",
    "2": "balanced",
    "3": "quality",
    "fast": "fast",
    "balanced": "balanced",
    "quality": "quality"
}


def main():
    video_path = input("Add meg a videó elérési útját: ").strip()
    profile_input = input(
        "Valaszd ki a profilt [1=fast, 2=balanced, 3=quality] (Enter=fast): "
    ).strip().lower()
    enhancement_profile = PROFILE_CHOICES.get(profile_input, "fast")

    logger.info("Kivalasztott profil: %s", enhancement_profile)

    processor = VideoProcessor(enhancement_profile=enhancement_profile)
    processor.process(video_path)


if __name__ == "__main__":
    main()