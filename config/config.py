from box import Box

frame_original_part = {
    'queue_length': 3,
    'resize': (741, 551)
}

frame_roi_part = {
    "set_ROI": {
        "roi_x": [85, 100],
        "roi_y": [0, 100],
        # 'roi_x': [25, 75],
        # "roi_y": [50, 100],
        "rectangle": False,
        "return_gray": True,
    },
    "renew_roi_background": {"cool_down": 2},
    "queue_frame_roi_enqueue": {},
    "queue_video_difference_enqueue": {"binary_threshold": 40, "queue_length": 2},
    "queue_video_difference_dequeue": {"threshold": 0.5},
    "queue_difference_enqueue": {"queue_length": 3},
    "queue_difference_dequeue": {"queue_length_sec": 0.5},
}

cycle_part = {
    "cycle_judge": {"cycle_cooldown": 10, "difference_threshold": 0.5},
    "cycle_sustain_set": {"cycle_sustain_time": 4},
}

inference_part = {
    "on_cycle_inference": {"inference_start_cycle": 1,
                           'extract': False},
    "inference_switch": {"n_inference_frame": 3, "switch_threshold": 0.05},
}

abnormal_part = {
    "abnormal_judge": {"abnormal_threshold": 0.95},
    "abnormal_alarm": {"patient": 5},
}

# Boxing -> can use as .
frame_original_part = Box(frame_original_part)
frame_roi_part = Box(frame_roi_part)
cycle_part = Box(cycle_part)
inference_part = Box(inference_part)
abnormal_part = Box(abnormal_part)
