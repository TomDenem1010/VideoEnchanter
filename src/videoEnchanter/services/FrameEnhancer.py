import cv2


class FrameEnhancer:

    def __init__(self, profile: str = "fast"):
        self.profile = profile
        self.clahe = cv2.createCLAHE(
            clipLimit=1.8,
            tileGridSize=(8, 8)
        )

    def _resize_for_quality_denoise(self, frame):
        height, width = frame.shape[:2]
        scaled_width = max(640, width // 2)
        scaled_height = max(360, height // 2)

        if scaled_width >= width or scaled_height >= height:
            return frame, None

        denoise_size = (scaled_width, scaled_height)
        resized = cv2.resize(frame, denoise_size, interpolation=cv2.INTER_AREA)
        return resized, (width, height)

    def _denoise_quality(self, frame):
        resized_frame, original_size = self._resize_for_quality_denoise(frame)
        denoised = cv2.fastNlMeansDenoisingColored(
            resized_frame,
            None,
            3,
            3,
            7,
            15
        )

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
        return cv2.addWeighted(
            frame,
            1.12,
            cv2.GaussianBlur(frame, (0, 0), 1.1),
            -0.12,
            0
        )

    def _denoise(self, frame):
        if self.profile == "quality":
            return self._denoise_quality(frame)

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