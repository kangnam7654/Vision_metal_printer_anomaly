import cv2

from utils.common.HongsModule import HongsModule
from utils.common.project_paths import GetProjectPath
import yaml


def load_config():
    config_path = GetProjectPath().get_project_root('config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.full_load(f)
    return config

config = load_config()

class HongsProcess(HongsModule):
    def __init__(self, video_file_path, roi_interval=1):
        super().__init__(video_file_path, roi_interval)

    def frame_original_part(self,
                            frame,
                            queue_length=config.frame_original_part.queue_length,
                            size=config.frame_original_part.resize):
        """
        원본 영상을 리사이즈하여 queue 에 enqueue 와 dequeue 를 시행합니다.
        :param frame:
        :param queue_length:
        :return:
        """
        frame_resize = self.resize_frame(frame, size)
        self.queue_frame.append(frame_resize)
        if len(self.queue_frame) >= queue_length:
            frame_resize = self.queue_frame.pop(0)
        return frame_resize

    def frame_roi_part(
        self,
        frame_resize,
        roi_x=config.frame_roi_part.set_ROI.roi_x,
        roi_y=config.frame_roi_part.set_ROI.roi_y,
        rectangle=config.frame_roi_part.set_ROI.rectangle,
        return_gray=config.frame_roi_part.set_ROI.return_gray,
        binary_threshold=config.frame_roi_part.queue_video_difference_enqueue.binary_threshold,
        cool_down=config.frame_roi_part.renew_roi_background.cool_down,
        enqueue_queue_length=config.frame_roi_part.queue_video_difference_enqueue.queue_length,
        dequeue_threshold=config.frame_roi_part.queue_video_difference_dequeue.threshold,
        queue_length=config.frame_roi_part.queue_difference_enqueue.queue_length,
        queue_length_sec=config.frame_roi_part.queue_difference_dequeue.queue_length_sec
    ):

        # cycle 판정 roi 설정
        frame_roi = self.set_ROI(
            frame_resize,
            roi_x=roi_x,
            roi_y=roi_y,
            rectangle=rectangle,
            return_gray=return_gray,
        )
        # TODO for test 삭제
        # f = frame_roi.copy()
        # f = cv2.resize(f, (224, 224))
        # cv2.imshow('roi', f)


        self.renew_roi_background(frame_roi, cool_down=cool_down)
        self.queue_frame_roi_enqueue(frame_roi=frame_roi)
        self.queue_video_difference_enqueue(binary_threshold=binary_threshold,
                                            queue_length=enqueue_queue_length)
        binary_difference = self.queue_video_difference_dequeue(
            frame_roi, queue_length=queue_length
        )
        self.queue_difference_enqueue(
            binary_difference=binary_difference, threshold=dequeue_threshold
        )
        self.queue_difference_dequeue(queue_length_sec=queue_length_sec)
        return binary_difference

    def show_part(self, binary_difference, frame_resize):
        """
        영상 재생부
        :param binary_difference: ROI 이진화 프레임
        :param frame_resize: 리사이즈 프레임
        """
        cv2.imshow("binary difference", binary_difference)
        cv2.imshow("original video", frame_resize)
        cv2.waitKey(self.key)

    def cycle_part(
        self,
        cycle_cool_down=config.cycle_part.cycle_judge.cycle_cooldown,
        difference_threshold=config.cycle_part.cycle_judge.difference_threshold,
        cycle_sustain_time=config.cycle_part.cycle_sustain_set.cycle_sustain_time,
    ):
        self.cycle_judge(
            cycle_cool_down=cycle_cool_down,
            difference_threshold=difference_threshold,
        )
        self.cycle_sustain_set(cycle_sustain_time=cycle_sustain_time)

    def inference_part(
        self,
        frame_resize,
        binary_difference,
        inference_start_cycle=config.inference_part.on_cycle_inference.inference_start_cycle,
        n_inference_frame=config.inference_part.inference_switch.n_inference_frame,
        switch_threshold=config.inference_part.inference_switch.switch_threshold,
        extract = config.inference_part.on_cycle_inference.extract
    ):
        self.inference_switch(
            binary_difference,
            n_inference_frame=n_inference_frame,
            switch_threshold=switch_threshold,
        )

        self.on_cycle_inference(frame_resize, inference_start_cycle=inference_start_cycle, extract=extract)

    def abnormal_part(self,
                      abnormal_threshold=config.abnormal_part.abnormal_judge.abnormal_threshold,
                      patient=config.abnormal_part.abnormal_alarm.patient):
        """

        :param abnormal_threshold:
        :param patient:
        :return:
        """
        self.abnormal_judge(abnormal_threshold=abnormal_threshold)
        self.abnormal_alarm(patient=patient)


if __name__ == "__main__":
    pass
