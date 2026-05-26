import cv2
import numpy as np


class FrameEnhancer:

    def __init__(self, profile: str = "fast"):
        self.profile = profile
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        self.sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

    def _denoise(self, frame):
        if self.profile == "quality":
            return cv2.fastNlMeansDenoisingColored(
                frame,
                None,
                5,
                5,
                7,
                21
            )

        if self.profile == "balanced":
            return cv2.bilateralFilter(
                frame,
                7,
                35,
                35
            )

        return cv2.GaussianBlur(frame, (3, 3), 0)

    def enhance(self, frame):
        denoised = self._denoise(frame)

        # CLAHE kontrasztjavítás
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        cl = self.clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        contrast = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        sharpened = cv2.filter2D(contrast, -1, self.sharpen_kernel)

        return sharpened