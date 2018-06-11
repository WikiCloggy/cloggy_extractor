import cv2
import numpy as np
from common import util

class cloggy_extractor:
    def __init__(self, filter_kernal_size=5, filter_iter_number=3, min_patch_size=12, max_patch_size=24):
        self.filter_kernal_size = filter_kernal_size
        self.filter_iter_number = filter_iter_number
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.version = '3.21'

    def delete_background(self, img, rect:tuple, skip_pixel=6, marker_size=8, bg_threshold=0.25, fg_threshold=0.25):
        print(marker_size, skip_pixel, bg_threshold, fg_threshold)
        height, width = img.shape[:2]
        if width != 640 and height != 640:
            img = self.optimze_image_size(img)
        _img = self.apply_filter(img)

        mask = np.zeros(img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(_img, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        #b, g, r = cv2.split(_img)
        #g = g * 2
        #marked_img = cv2.merge((b, g, r))

        #bg_color_list = self.extract_color_around_rect(_img, mask, rect)
        #fg_color_list = self.extract_foreground_color(_img, mask, rect)

        fg_color_list, bg_color_list = self.extract_color_list(img, mask, rect)

        #self.mark_image(_img, mask, rect, bg_color_list, marker_size, skip_pixel, bg_threshold)
        #self.mark_image(_img, mask, rect, fg_color_list, marker_size, skip_pixel, fg_threshold, 1)
        mask = self.mark_mask(mask, img, rect,
                              fg_color_list=fg_color_list, bg_color_list=bg_color_list,
                              marker_size=marker_size, skip_pixel=skip_pixel,
                              bg_threshold=bg_threshold, fg_threshold=fg_threshold)
        cv2.grabCut(_img, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

        kernal_size = marker_size
        if kernal_size % 2 == 0:
            kernal_size += 1
        kernal = np.ones((kernal_size, kernal_size), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernal)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernal)

        return mask2

    def apply_filter(self, img):
        img_copy = img.copy()
        for iteration in range(self.filter_iter_number):
            img_copy = cv2.medianBlur(img_copy, self.filter_kernal_size)
        return img_copy

    def extract_color_around_rect(self, img, mask, rect):
        color_list = []
        height, width = img.shape[:2]
        rect_x, rect_y, rect_width, rect_height = rect

        patch_size = round(min(height / 10, width / 10))
        patch_size = max(self.min_patch_size, patch_size)
        patch_size = min(patch_size, self.max_patch_size)

        if rect_x > 0:
            if rect_x - patch_size >= 0:
                _patch_size = patch_size
            else:
                _patch_size = rect_x

            startX = rect_x - _patch_size
            endX = rect_x
            startY = rect_y
            endY = rect_y + rect_height

            self.get_average_color(img, mask, _patch_size, startX, endX, startY, endY, color_list, except_black=True)

        if width > rect_x + rect_width:
            if width - patch_size >= rect_x + rect_width:
                _patch_size = patch_size
            else:
                _patch_size = width - (rect_x + rect_width)

            startX = rect_x + rect_width
            endX = startX + _patch_size
            startY = rect_y
            endY = startY + rect_height

            self.get_average_color(img, mask, _patch_size, startX, endX, startY, endY, color_list, except_black=True)

        if rect_y > 0:
            if rect_y - patch_size >= 0:
                _patch_size = patch_size
            else:
                _patch_size = rect_y

            startX = 0
            endX = width
            startY = rect_y - _patch_size
            endY = rect_y

            self.get_average_color(img, mask, _patch_size, startX, endX, startY, endY, color_list, vertical=False, except_black=True)

        if height > rect_y + rect_height:
            if height - patch_size >= rect_y + rect_height:
                _patch_size = patch_size
            else:
                _patch_size = height - (rect_y + rect_height)

            startX = 0
            endX = width
            startY = rect_y + rect_height
            endY = startY + _patch_size

            self.get_average_color(img, mask, _patch_size, startX, endX, startY, endY, color_list, vertical=False, except_black=True)

        return np.array(color_list)

    def extract_color_list(self, img, mask, rect):
        rectX, rectY, rectWidth, rectHeight = rect
        fg_color_list = []
        bg_color_list = []
        init_color = self.get_init_color(img)

        fg_average_color = init_color
        bg_average_color = init_color

        patch_size = round(min(rectHeight / 10, rectWidth / 10))
        patch_size = max(self.min_patch_size, patch_size)
        patch_size = min(patch_size, self.max_patch_size)

        startX, startY = (rectX, rectY)
        while (startY < rectY + rectHeight - patch_size) and (startX < rectX + rectWidth - patch_size):
            fg_n = 0
            bg_n = 0
            for y in range(startY, startY + patch_size):
                for x in range(startX, startX + patch_size):
                    if mask[y, x] == 0 or mask[y, x] == 2:
                        bg_average_color += img[y, x]
                        bg_n += 1
                    else:
                        fg_average_color += img[y, x]
                        fg_n += 1
            if fg_n > 1:
                fg_average_color = fg_average_color / fg_n
            if bg_n > 1:
                bg_average_color = bg_average_color / bg_n

            fg_std = init_color
            bg_std = init_color

            for y in range(startY, startY + patch_size):
                for x in range(startX, startX + patch_size):
                    if mask[y, x] == 0 or mask[y, x] == 2:
                        bg_std += (img[y, x] - bg_average_color)**2
                    else:
                        fg_std += (img[y, x] - fg_average_color)**2

            if fg_n > 1:
                fg_std = fg_std / (fg_n - 1)
                fg_std = np.sqrt(fg_std)
                #if np.mean(fg_average_color >= 250).all():
                 #   break
                info = {'mean' : fg_average_color, 'std' : fg_std}
                fg_color_list.append(info)
            if bg_n > 1:
                bg_std = bg_std / (bg_n - 1)
                bg_std = np.sqrt(bg_std)
                #if np.mean(bg_average_color <= 3).all():
                 #   break
                info = {'mean' : bg_average_color, 'std' : bg_std}
                bg_color_list.append(info)

            fg_average_color = init_color
            bg_average_color = init_color

            fg_n = 0
            bg_n = 0
            startX += patch_size
            startY += patch_size
        return fg_color_list, bg_color_list

    def extract_foreground_color(self, img, mask, rect):
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
            self.get_average_color(img, mask, self.min_patch_size,
                                   centerX - space, centerX + space,
                                   centerY - round(self.min_patch_size / 2), centerY + round(self.min_patch_size / 2),
                                   color_list, vertical=False, bg=False, except_white=True)
        else:
            space = round(height / 8)
            self.get_average_color(img, self.min_patch_size,
                                   centerX - round(self.min_patch_size / 2), centerX + round(self.min_patch_size / 2),
                                   centerY - space, centerY + space,
                                   color_list, vertical=True, bg=False, except_white=True)
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
                    diff = abs(color_list[i] - src[y, x])
                    #print(x, y, distance, np.exp(distance))
                    """
                    normalization = 0
                    for j in range(chanel):
                        c1 = color_list[i][j] + 1
                        c2 = src[y, x][j] + 1
                        normalization += max(c1, c2) / min(c1, c2)
                    normalization = normalization / chanel
                    normalization = np.exp(normalization)
                    if normalization < threshold:"""
                    if (diff < threshold).all():
                        #print(x, y, normalization, threshold, mark_color)
                        dst = cv2.circle(dst, (x, y), marker_size, mark_color, -1)

        return dst

    def mark_mask(self, mask, img, rect, bg_color_list, fg_color_list, marker_size=8, skip_pixel=6, bg_threshold=2.5, fg_threshold=2.5):
        rect_x, rect_y, rect_width, rect_height = rect
        for y in range(rect_y, rect_y + rect_height, skip_pixel):
            for x in range(rect_x, rect_x + rect_width, skip_pixel):
                if mask[y, x] >= 2:
                    for info in bg_color_list:
                        z = img[y, x] - info['mean']
                        z = z / info['std']

                        if (abs(z) < bg_threshold).all():
                            mask = cv2.circle(mask, (x, y), marker_size, 0, -1)
                elif mask[y, x] == 3:
                    for info in fg_color_list:
                        z = img[y, x] - info['mean']
                        z = z / info['std']

                        if (abs(z) < fg_threshold).all():
                            mask = cv2.circle(mask, (x, y), marker_size, 1, -1)
        return mask

    def get_average_color(self, img, mask, _patch_size, startX, endX, startY, endY, array, vertical=True, bg=True, except_white=False, except_black=False):
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
        n = 0
        for a in range(startA, endA):
            for b in range(startB, endB):
                if vertical:
                    if bg:
                        if mask[a, b] == 2 or mask[a, b] == 0:
                            average_color += img[a, b]
                            n += 1
                    else:
                        if mask[a, b] == 3 or mask[a, b] == 1:
                            average_color += img[a, b]
                            n += 1
                else:
                    if bg:
                        if mask[b][a] == 2:
                            average_color += img[b, a]
                            n += 1
                    else:
                        if mask[b][a] == 3:
                            average_color += img[b, a]
                            n += 1
            step += 1

            if step == _patch_size or a == endA - 1:
                #average_color = average_color / (step * _patch_size)
                #print(print(step * _patch_size))
                if n != 0:
                    average_color = np.round(average_color / n).astype('uint8')
                    if np.mean(average_color) >= 245 and except_white:
                        pass
                    elif np.mean(average_color) <= 3 and except_black:
                        pass
                    else:
                        array.append(average_color)
                average_color = _init_color
                step = 0
                n = 0

    def get_init_color(self, img):
        if len(img.shape) == 3:
            init_color = [0, 0, 0]
        else:
            init_color = 0
        return init_color

    def optimze_image_size(self, img):
        print('resized')
        height, width = img.shape[:2]
        if width > height:
            ratio = 640 / width
            resize_width = 640
            resize_height = round(height * ratio)
        else:
            ratio = 640 / height
            resize_height = 640
            resize_width = round(width * ratio)
        img = util.resizeImage(img, (resize_width, resize_height), (0, 0, width, height))  # #show_image(img)
        return img
