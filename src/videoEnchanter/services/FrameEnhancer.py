import cv2


class FrameEnhancer:

    def __init__(self, profile: str = "fast"):
        self.profile = profile
        self.clahe = cv2.createCLAHE(
            clipLimit=1.35,
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

    def _denoise_quality(self, frame):
        resized_frame, original_size = self._resize_for_quality_denoise(frame)
        denoised = cv2.bilateralFilter(
            resized_frame,
            7,
            30,
            30
        )

        denoised = cv2.GaussianBlur(denoised, (0, 0), 0.35)

        if original_size is None:
            return denoised

        return cv2.resize(denoised, original_size, interpolation=cv2.INTER_LINEAR)

    def _adjust_contrast(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self.clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _sharpen(self, frame):
        if self.profile == "quality":
            return cv2.addWeighted(
                frame,
                1.06,
                cv2.GaussianBlur(frame, (0, 0), 1.15),
                -0.06,
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