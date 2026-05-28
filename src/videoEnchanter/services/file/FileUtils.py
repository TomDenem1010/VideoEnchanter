from datetime import datetime
from pathlib import Path


def buildOutputPath(video_path: str, processing_type: str) -> str:
    path = Path(video_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return str(
        path.parent / f"{path.stem}_{processing_type}_{timestamp}{path.suffix}"
    )


def buildTemporaryOutputPath(output_path: str) -> str:
    path = Path(output_path)

    return str(
        path.parent / f"{path.stem}_silent{path.suffix}"
    )