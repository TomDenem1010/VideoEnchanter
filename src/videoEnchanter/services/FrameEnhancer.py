import cv2
import numpy as np


class FrameEnhancer:

    def enhance(self, frame):
        # Zajszűrés
        denoised = cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            5,
            5,
            7,
            21
        )

        # CLAHE kontrasztjavítás
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        contrast = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # Élesítés
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        sharpened = cv2.filter2D(contrast, -1, kernel)

        return sharpened