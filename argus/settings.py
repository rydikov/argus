SILENT_TIME = 1800 # 30 minutes

# Disable telegram alerting on silent time after detection
# Last time when notification about detected frame has been sent
notification_throttlers = {}

# Last time when frame has been saved
save_throttlers = {}

# List for frames to be sent after an external signal
send_frames_after_signal = []


# Protection against false detections
multi_hit_confirmations = {}