import cv2
import numpy as np

from videoEnchanter.services.frameEnhancer.FrameEnhancer import FrameEnhancer


class QualityFrameEnhancer(FrameEnhancer):

    def _restore_detail(self, frame):
        # Create a soft, upscale-like detail boost without changing resolution.
        upscaled = cv2.resize(frame, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
        reconstructed = cv2.resize(
            upscaled,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_AREA
        )

        detail_layer = cv2.subtract(frame, reconstructed)
        boosted = cv2.addWeighted(frame, 1.0, detail_layer, 0.7, 0)
        return np.clip(boosted, 0, 255).astype(frame.dtype)

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
        brightness = self._get_frame_brightness(frame)

        if brightness >= 0.62:
            return self._apply_clahe(frame, 1.15)

        if brightness >= 0.48:
            return self._apply_clahe(frame, 1.35)

        return self._apply_clahe(frame, 1.55)

    def _sharpen(self, frame):
        brightness = self._get_frame_brightness(frame)

        if brightness >= 0.62:
            return self._apply_unsharp_mask(frame, 1.09, 0.95)

        if brightness >= 0.48:
            return self._apply_unsharp_mask(frame, 1.12, 1.05)

        return self._apply_unsharp_mask(frame, 1.14, 1.15)

    def enhance(self, frame):
        denoised = self._denoise_quality(frame)
        restored = self._restore_detail(denoised)
        contrast = self._adjust_contrast(restored)
        return self._sharpen(contrast)