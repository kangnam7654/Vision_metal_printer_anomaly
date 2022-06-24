from utils.common.HongsProcess import HongsProcess
import time


class SAMPLE(HongsProcess):
    def __init__(self, video_file_path, roi_interval=1):
        super().__init__(video_file_path=video_file_path, roi_interval=roi_interval)

    def sample_flow(self, play_speed=1):
        cap = self.cap
        # assert -> if raise
        if not cap.isOpened():
            raise Exception("video not opened")

        # 시간 측정 시작부 #
        time_start = time.time()
        ####################
        while True:
            retval, frame = cap.read()
            # assert -> if raise
            if not retval:
                raise Exception(f'return_value: {retval}')
            self.frame_counter += 1

            frame_resize = self.frame_original_part(frame)
            binary_difference = self.frame_roi_part(frame_resize)
            self.show_part(
                frame_resize=frame_resize, binary_difference=binary_difference
            )
            self.cycle_part()
            self.inference_part(
                frame_resize=frame_resize, binary_difference=binary_difference
            )
            self.abnormal_part()

            time_end = time.time()
            self.current_fps = self.calculate_current_fps(time_start, time_end)
            time_start = time_end
            aim_fps = self.original_fps * play_speed
            self.control_waitkey(aim_fps)


if __name__ == '__main__':
    file = 'D:\project\Hongsworks\src\data\source_videos\\15_00_56.avi'
    sample = SAMPLE(file, roi_interval=2)
    sample.sample_flow(play_speed=2)