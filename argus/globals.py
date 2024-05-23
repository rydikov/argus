from datetime import timedelta

SILENT_TIME = timedelta(minutes=30)

# No telegram alerting on silent time after detection
detected_frame_notification_time = {}

# Last time when frame has been saved
last_frame_save_time = {}

# List for frames to be sent after an external signal
send_frames_after_signal = []