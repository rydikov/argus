from datetime import timedelta

SILENT_TIME = timedelta(minutes=30)

last_detection = {}

# Last time when frame has been saved
last_frame_save_time = {}

# No telegram alerting on silent time after detection
silent_notify_until_time = {}

# List for frames to be sent after an external signal
send_frames_after_signal = []