import cv2
import numpy as np


class FrameEnhancer:

    def __init__(self, profile: str = "fast"):
        self.profile = profile
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

    def _denoise_quality(self, frame):
        resized_frame, original_size = self._resize_for_quality_denoise(frame)
        brightness = self._get_frame_brightness(resized_frame)

        if brightness >= 0.62:
            denoised = cv2.bilateralFilter(
                resized_frame,
                7,
                24,
                24
            )
            denoised = cv2.GaussianBlur(denoised, (0, 0), 0.25)
        elif brightness >= 0.48:
            denoised = cv2.bilateralFilter(
                resized_frame,
                9,
                34,
                34
            )
            denoised = cv2.GaussianBlur(denoised, (0, 0), 0.4)
        else:
            denoised = cv2.bilateralFilter(
                resized_frame,
                11,
                48,
                48
            )
            denoised = cv2.GaussianBlur(denoised, (0, 0), 0.65)

        if original_size is None:
            return denoised

        return cv2.resize(denoised, original_size, interpolation=cv2.INTER_LINEAR)

    def _adjust_contrast(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        if self.profile == "quality":
            brightness = float(np.mean(l_channel)) / 255.0

            if brightness >= 0.62:
                clahe = cv2.createCLAHE(clipLimit=1.15, tileGridSize=(8, 8))
            elif brightness >= 0.48:
                clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
            else:
                clahe = self.clahe

            l_channel = clahe.apply(l_channel)
        else:
            l_channel = self.clahe.apply(l_channel)

        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _sharpen(self, frame):
        if self.profile == "quality":
            brightness = self._get_frame_brightness(frame)

            if brightness >= 0.62:
                amount = 1.07
                blur_sigma = 1.0
            elif brightness >= 0.48:
                amount = 1.09
                blur_sigma = 1.15
            else:
                amount = 1.11
                blur_sigma = 1.3

            return cv2.addWeighted(
                frame,
                amount,
                cv2.GaussianBlur(frame, (0, 0), blur_sigma),
                -(amount - 1.0),
                0
            )

        return cv2.addWeighted(
            frame,
            1.03,
            cv2.GaussianBlur(frame, (0, 0), 0.9),
            -0.03,
            0
        )

    def _denoise(self, frame):
        if self.profile == "quality":
            return self._denoise_quality(frame)

        return cv2.GaussianBlur(frame, (5, 5), 0.9)

    def enhance(self, frame):
        denoised = self._denoise(frame)
        contrast = self._adjust_contrast(denoised)
        return self._sharpen(contrast)