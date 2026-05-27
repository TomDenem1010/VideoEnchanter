import cv2
import numpy as np


class FrameEnhancer:

    def __init__(self, profile: str = "fast"):
        self.profile = profile
        self.clahe = cv2.createCLAHE(
            clipLimit=1.8,
            tileGridSize=(8, 8)
        )
        self.sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

    def _adjust_contrast(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self.clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _sharpen(self, frame):
        return cv2.addWeighted(
            frame,
            1.12,
            cv2.GaussianBlur(frame, (0, 0), 1.1),
            -0.12,
            0
        )

    def _denoise(self, frame):
        if self.profile == "quality":
            return cv2.fastNlMeansDenoisingColored(
                frame,
                None,
                3,
                3,
                7,
                15
            )

        if self.profile == "balanced":
            return cv2.bilateralFilter(
                frame,
                5,
                28,
                28
            )

        return cv2.bilateralFilter(
            frame,
            3,
            18,
            18
        )

    def enhance(self, frame):
        denoised = self._denoise(frame)
        contrast = self._adjust_contrast(denoised)
        return self._sharpen(contrast)