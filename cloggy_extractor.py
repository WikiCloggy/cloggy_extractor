import cv2
import numpy as np

class cloggy_extractor:
    def __init__(self, filter_kernal_size=3, filter_iter_number=30, min_patch_size=6, max_patch_size=18):
        self.filter_kernal_size = filter_kernal_size
        self.filter_iter_number = filter_iter_number
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size

    def delete_background(self, img, rect:tuple, skip_pixel=6, marker_size=8, bg_threshold=2.5, fg_threshold=2.5):
        _img = self.apply_filter(img)
        mask = np.zeros(img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(_img, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)

        bg_color_list = self.extract_color_around_rect(_img, rect)
        fg_color_list = self.extract_foreground_color(_img, rect)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        self.mark_image(_img, mask, rect, bg_color_list, marker_size, skip_pixel, bg_threshold)
        self.mark_image(_img, mask, rect, fg_color_list, marker_size, skip_pixel, fg_threshold, 1)

        cv2.grabCut(_img, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

        return mask2

    def apply_filter(self, img):
        img_copy = img.copy()
        for iteration in range(self.filter_iter_number):
            img_copy = cv2.medianBlur(img_copy, self.filter_kernal_size)
        return img_copy

    def extract_color_around_rect(self, img, rect):
        color_list = []
        height, width = img.shape[:2]
        rect_x, rect_y, rect_width, rect_height = rect

        patch_size = round(min(height / 10, width / 10))
        patch_size = max(self.min_patch_size, patch_size)
        patch_size = min(patch_size, self.max_patch_size)

        if len(img.shape) == 3:
            init_color = [0, 0, 0]
        else:
            init_color = 0

        if rect_x > 0:
            if rect_x - patch_size >= 0:
                _patch_size = patch_size
            else:
                _patch_size = rect_x

            startX = rect_x - _patch_size
            endX = rect_x
            startY = rect_y
            endY = rect_y + rect_height

            self.get_average_color(img, _patch_size, startX, endX, startY, endY, color_list)

        if width > rect_x + rect_width:
            if width - patch_size >= rect_x + rect_width:
                _patch_size = patch_size
            else:
                _patch_size = width - (rect_x + rect_width)

            startX = rect_x + rect_width
            endX = startX + _patch_size
            startY = rect_y
            endY = startY + rect_height

            self.get_average_color(img, _patch_size, startX, endX, startY, endY, color_list)

        if rect_y > 0:
            if rect_y - patch_size >= 0:
                _patch_size = patch_size
            else:
                _patch_size = rect_y

            startX = 0
            endX = width
            startY = rect_y - _patch_size
            endY = rect_y

            self.get_average_color(img, _patch_size, startX, endX, startY, endY, color_list, False)

        if height > rect_y + rect_height:
            if height - patch_size >= rect_y + rect_height:
                _patch_size = patch_size
            else:
                _patch_size = height - (rect_y + rect_height)

            startX = 0
            endX = width
            startY = rect_y + rect_height
            endY = startY + _patch_size

            self.get_average_color(img, _patch_size, startX, endX, startY, endY, color_list, False)

        return np.array(color_list)

    def extract_foreground_color(self, img, rect):
        x, y, width, height = rect
        centerX = x + round(width / 2)
        centerY = y + round(height / 2)
        color_list = []

        #space = round(self.min_patch_size / 2)

        """
        self.get_average_color(img, self.min_patch_size,
                               centerX - self.min_patch_size - space, centerX + self.min_patch_size + space,
                               centerY - space, centerY + space,
                               color_list, False)
        self.get_average_color(img, self.min_patch_size,
                               centerX - self.min_patch_size - space, centerX + self.min_patch_size + space,
                               centerY - self.min_patch_size - space, centerY - self.min_patch_size + space,
                               color_list, False)
        self.get_average_color(img, self.min_patch_size,
                               centerX - self.min_patch_size - space, centerX + self.min_patch_size + space,
                               centerY + self.min_patch_size - space, centerY + self.min_patch_size + space,
                               color_list, False)
        """
        if width > height:
            space = round(width / 8)
            self.get_average_color(img, self.min_patch_size,
                                   centerX - space, centerX + space,
                                   centerY - round(self.min_patch_size / 2), centerY + round(self.min_patch_size / 2),
                                   color_list, False, True)
        else:
            space = round(height / 8)
            self.get_average_color(img, self.min_patch_size,
                                   centerX - round(self.min_patch_size / 2), centerX + round(self.min_patch_size / 2),
                                   centerY - space, centerY + space,
                                   color_list, True, True)
        return np.array(color_list)

    def mark_image(self, src, dst, rect, color_list, marker_size=6, skip_pixel=6, threshold=3, mark_color=0):
        rect_x, rect_y, rect_width, rect_height = rect

        try:
            chanel = src.shape[2]
        except:
            chanel = 1
        for y in range(rect_y, rect_y + rect_height, skip_pixel):
            for x in range(rect_x, rect_x + rect_width, skip_pixel):
                for i in range(color_list.shape[0]):
                    diff_mean = abs(color_list[i] - src[y, x])
                    diff_mean = abs(np.mean(diff_mean))
                    """
                    normalization = 0
                    for j in range(chanel):
                        c1 = color_list[i][j] + 1
                        c2 = src[y, x][j] + 1
                        normalization += max(c1, c2) / min(c1, c2)
                    normalization = normalization / chanel
                    normalization = np.exp(normalization)
                    if normalization < threshold:"""
                    if np.exp(diff_mean) < threshold:
                        #print(x, y, normalization, threshold, mark_color)
                        dst = cv2.circle(dst, (x, y), marker_size, mark_color, -1)

        return dst

    def get_average_color(self, img, _patch_size, startX, endX, startY, endY, array, vertical=True, except_white=False):
        _init_color = self.get_init_color(img)

        if _patch_size < self.min_patch_size:
            return
        average_color = _init_color

        if vertical:
            startA = startY
            endA = endY
            startB = startX
            endB = endX

        else:
            startA = startX
            endA = endX
            startB = startY
            endB = endY

        step = 0
        for a in range(startA, endA):
            for b in range(startB, endB):
                if vertical:
                    average_color += img[a, b]
                else:
                    average_color += img[b, a]

            step += 1
            if step == _patch_size or a == endA - 1:
                #average_color = average_color / (step * _patch_size)
                #print(print(step * _patch_size))
                average_color = np.round(average_color / (step * _patch_size)).astype('uint8')
                if np.mean(average_color) >= 245 and except_white:
                    print(average_color)
                    pass
                else:
                    array.append(average_color)
                average_color = _init_color
                step = 0

    def get_init_color(self, img):
        if len(img.shape) == 3:
            init_color = [0, 0, 0]
        else:
            init_color = 0
        return init_color
