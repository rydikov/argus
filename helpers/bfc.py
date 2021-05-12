import os
import collections


class BadFrameChecker(object):
    
    def __init__(self):
        self.store_images = 5
        self.last_image_sizes = collections.deque([], self.store_images)
        self.deviation_percent = 60

    def is_image_size_less_avg_size(self, image_size):
        avg_image_size = sum(self.last_image_sizes)/self.store_images
        return (image_size/avg_image_size)*100 < self.deviation_percent

    def is_bad(self, image_path):
        image_size = os.path.getsize(image_path)
        self.last_image_sizes.appendleft(image_size)

        if len(self.last_image_sizes) < self.store_images:
            return False
        
        return self.is_image_size_less_avg_size(image_size)