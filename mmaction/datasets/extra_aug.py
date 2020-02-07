import mmcv
import numpy as np


class PhotoMetricDistortion(object):
    def __init__(self, brightness_range=None, contrast_range=None, saturation_range=None,
                 hue_delta=None, noise_sigma=None, color_scale=None):
        self.brightness_lower, self.brightness_upper =\
            brightness_range if brightness_range is not None else (None, None)
        self.contrast_lower, self.contrast_upper =\
            contrast_range if contrast_range is not None else (None, None)
        self.saturation_lower, self.saturation_upper =\
            saturation_range if saturation_range is not None else (None, None)
        self.hue_delta = hue_delta if hue_delta is not None else None
        self.noise_sigma = noise_sigma if noise_sigma is not None else None
        self.color_scale_lower, self.color_scale_upper = color_scale if color_scale is not None else (None, None)

    @property
    def _with_brightness(self):
        return self.brightness_lower is not None and self.brightness_upper is not None

    @property
    def _with_contrast(self):
        return self.contrast_lower is not None and self.contrast_upper is not None

    @property
    def _with_saturation(self):
        return self.saturation_lower is not None and self.saturation_upper is not None

    @property
    def _with_hue(self):
        return self.hue_delta is not None

    @property
    def _with_noise(self):
        return self.noise_sigma is not None

    @property
    def _with_color_scale(self):
        return self.color_scale_lower is not None and self.color_scale_upper is not None

    @staticmethod
    def _augm(img, brightness_delta, contrast_mode, contrast_alpha, saturation_alpha,
              hue_delta, noise_sigma, color_scales):
        def _clamp_image(_img):
            _img[_img < 0.0] = 0.0
            _img[_img > 255.0] = 255.0
            return _img

        img = img.astype(np.float32)

        # random brightness
        if brightness_delta is not None:
            img += brightness_delta
            img = _clamp_image(img)

        # random contrast
        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        if contrast_mode == 1:
            if contrast_alpha is not None:
                img *= contrast_alpha
                img = _clamp_image(img)

        # convert color from BGR to HSV
        if saturation_alpha is not None or hue_delta is not None:
            img = mmcv.bgr2hsv(img / 255.)

        # random saturation
        if saturation_alpha is not None:
            img[:, :, 1] *= saturation_alpha
            img[:, :, 1][img[:, :, 1] > 1.0] = 1.0
            img[:, :, 1][img[:, :, 1] < 0.0] = 0.0

        # random hue
        if hue_delta is not None:
            img[:, :, 0] += hue_delta
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0

        # convert color from HSV to BGR
        if saturation_alpha is not None or hue_delta is not None:
            img = mmcv.hsv2bgr(img) * 255.

        # random contrast
        if contrast_mode == 0:
            if contrast_alpha is not None:
                img *= contrast_alpha
                img = _clamp_image(img)

        if color_scales is not None:
            img *= color_scales.reshape((1, 1, -1))

        # gaussian noise
        if noise_sigma is not None:
            img += np.random.normal(loc=0.0, scale=noise_sigma, size=img.shape)

        # clamp
        img = _clamp_image(img)

        return img.astype(np.uint8)

    def __call__(self, img_group):
        if self._with_brightness and np.random.randint(2):
            images_mean_brightness = [np.mean(img) for img in img_group]
            image_brightness = np.random.choice(images_mean_brightness)

            brightness_delta_limits = [self.brightness_lower - image_brightness,
                                       self.brightness_upper - image_brightness]
            if image_brightness < self.brightness_lower:
                brightness_delta_limits[0] = 0.0
            elif image_brightness > self.brightness_upper:
                brightness_delta_limits[1] = 0.0

            brightness_delta = np.random.uniform(brightness_delta_limits[0], brightness_delta_limits[1])
        else:
            brightness_delta = None

        contrast_mode = np.random.randint(2)
        contrast_alpha = np.random.uniform(self.contrast_lower, self.contrast_upper) \
            if self._with_contrast and np.random.randint(2) else None

        saturation_alpha = np.random.uniform(self.saturation_lower, self.saturation_upper) \
            if self._with_saturation and np.random.randint(2) else None

        hue_delta = np.random.uniform(-self.hue_delta, self.hue_delta)\
            if self._with_hue and np.random.randint(2) else None

        noise_sigma = np.random.uniform(self.noise_sigma[0], self.noise_sigma[1])\
            if self._with_noise and np.random.randint(2) else None

        color_scales = np.random.uniform(self.color_scale_lower, self.color_scale_upper, size=3)\
            if self._with_color_scale and np.random.randint(2) else None

        img_group = [self._augm(img, brightness_delta, contrast_mode, contrast_alpha,
                                saturation_alpha, hue_delta, noise_sigma, color_scales)
                     for img in img_group]

        return img_group
