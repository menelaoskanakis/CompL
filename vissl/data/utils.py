import numpy as np

class DenormalizeImage(object):
    """Denormalize image based on imagenet values
    Returns:
        img: Denormalized image
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        """
        self.mean = np.array(mean).reshape(-1, 1, 1)
        self.std = np.array(std).reshape(-1, 1, 1)

    def __call__(self, img):
        """
        """
        img = self.std * img + self.mean
        img = np.clip(img, 0, 1)
        return img

class VOCSegmentationMaskDecoder(object):
    """Decode predictions to semantic image
    """
    def __init__(self, n_classes=21):
        self.labels = np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )[0:n_classes]

    def __call__(self, label_mask):
        """
        Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
        Returns:
            rgb (np.ndarray): the resulting decoded color image.
        """
        # Initialize the segmentation map using the "void" colour
        r = np.ones_like(label_mask).astype(np.uint8) * 224
        g = np.ones_like(label_mask).astype(np.uint8) * 223
        b = np.ones_like(label_mask).astype(np.uint8) * 192

        for ind, label_colour in enumerate(self.labels):
            r[label_mask == ind] = label_colour[0]
            g[label_mask == ind] = label_colour[1]
            b[label_mask == ind] = label_colour[2]

        rgb = np.stack([r, g, b], axis=0) / 255.0
        return rgb
