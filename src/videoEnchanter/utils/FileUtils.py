from pathlib import Path


def buildOutputPath(video_path: str) -> str:
    path = Path(video_path)

    return str(
        path.parent / f"{path.stem}_enchanted{path.suffix}"
    )