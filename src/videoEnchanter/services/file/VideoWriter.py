import cv2


class VideoWriter:

    def create(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int
    ):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Nem sikerült létrehozni a videót: {output_path}"
            )

        return writer