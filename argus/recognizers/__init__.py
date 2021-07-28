

class Recognizer:

    def __init__(self, config):
        self.config = config
        self.init_network()

    def init_network(self, config):
        raise NotImplementedError

    def recognize(self, frame):
        raise NotImplementedError

    def split_and_recocnize(self, frame):
        h, w = frame.shape[0], frame.shape[1]  # e.g. 1080x1920

        # 960
        half_frame = int(w/2)

        # [120:1080, 0:960]
        left_frame = frame[h-half_frame:h, 0:half_frame]

        # [120:1080, 960:1920]
        right_frame = frame[h-half_frame:h, half_frame:w]

        left_frame_objects = self.recognize(left_frame)
        right_frame_objects = self.recognize(right_frame)

        for obj in left_frame_objects:
            obj['ymin'] += h - half_frame
            obj['ymax'] += h - half_frame

        for obj in right_frame_objects:
            obj['xmin'] += half_frame
            obj['ymin'] += h - half_frame
            obj['xmax'] += half_frame
            obj['ymax'] += h - half_frame

        return left_frame_objects + right_frame_objects
