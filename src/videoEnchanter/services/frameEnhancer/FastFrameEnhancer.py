import cv2

from videoEnchanter.services.frameEnhancer.FrameEnhancer import FrameEnhancer


class FastFrameEnhancer(FrameEnhancer):

    def enhance(self, frame):
        denoised = cv2.GaussianBlur(frame, (5, 5), 0.9)
        contrast = self._apply_clahe(denoised, 1.55)
        return self._apply_unsharp_mask(contrast, 1.03, 0.9)