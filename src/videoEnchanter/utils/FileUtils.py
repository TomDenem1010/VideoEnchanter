from pathlib import Path


def buildOutputPath(video_path: str) -> str:
    path = Path(video_path)

    return str(
        path.parent / f"{path.stem}_enchanted{path.suffix}"
    )


def buildTemporaryOutputPath(output_path: str) -> str:
    path = Path(output_path)

    return str(
        path.parent / f"{path.stem}_silent{path.suffix}"
    )