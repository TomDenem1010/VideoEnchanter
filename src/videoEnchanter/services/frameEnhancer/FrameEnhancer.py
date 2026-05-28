from abc import ABC, abstractmethod

import cv2
import numpy as np


class FrameEnhancer(ABC):

    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=1.55,
            tileGridSize=(8, 8)
        )

    def _resize_for_quality_denoise(self, frame):
        height, width = frame.shape[:2]
        scaled_width = max(960, width // 2)
        scaled_height = max(540, height // 2)

        if scaled_width >= width or scaled_height >= height:
            return frame, None

        denoise_size = (scaled_width, scaled_height)
        resized = cv2.resize(frame, denoise_size, interpolation=cv2.INTER_AREA)
        return resized, (width, height)

    def _get_frame_brightness(self, frame):
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, _a_channel, _b_channel = cv2.split(lab_frame)
        return float(np.mean(l_channel)) / 255.0

    def _apply_clahe(self, frame, clip_limit):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _apply_unsharp_mask(self, frame, amount, blur_sigma):
        return cv2.addWeighted(
            frame,
            amount,
            cv2.GaussianBlur(frame, (0, 0), blur_sigma),
            -(amount - 1.0),
            0
        )

    @abstractmethod
    def enhance(self, frame):
        pass