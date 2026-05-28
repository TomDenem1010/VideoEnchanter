import argparse
import logging
import multiprocessing
from services.VideoProcessor import VideoProcessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

PROFILE_CHOICES = {
    "1": "fast",
    "2": "quality",
    "fast": "fast",
    "quality": "quality"
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?")
    parser.add_argument(
        "--profile",
        default=None
    )
    args, _unknown_args = parser.parse_known_args()
    return args


def _resolve_video_path(video_path):
    if video_path:
        return video_path.strip()

    return input("Add meg a videó elérési útját: ").strip()


def _resolve_profile(profile):
    if profile:
        return PROFILE_CHOICES.get(profile.strip().lower(), "fast")

    profile_input = input(
        "Valaszd ki a profilt [1=fast, 2=quality] (Enter=fast): "
    ).strip().lower()
    return PROFILE_CHOICES.get(profile_input, "fast")


def main():
    args = _parse_args()
    video_path = _resolve_video_path(args.video_path)
    enhancement_profile = _resolve_profile(args.profile)

    logger.info("Kivalasztott profil: %s", enhancement_profile)

    processor = VideoProcessor(enhancement_profile=enhancement_profile)
    processor.process(video_path)
    input("Folyamat befejezve. Nyomj Entert a kilépéshez...")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()