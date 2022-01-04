import math
import os
import pickle
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn
import uuid
from models.VGG16 import VGG16
from utils.common.project_paths import GetProjectPath


class HongsModule:
    def __init__(self, video_file_path=0, roi_interval: int = 4):
        """홍스웍스의 이상 탐지에 필요한 모듈들을 모아놓은 클래스입니다.
        :param video_file_path: 비디오의 경로입니다. 경로에 비디오가 없을경우, 캠이 켜지게 됩니다.
        :param roi_interval: 몇 프레임마다 ROI 부분을 추출하는지 결정하는 변수입니다.
        """
        self.__paths = GetProjectPath()
        self.video_file = self.get_video_path(video=video_file_path)
        self.cap = cv2.VideoCapture(self.video_file)
        self.aspect_ratio = self.get_aspect_ratio()
        self.answer_vectors = self.get_answer_vectors()
        self.model = VGG16.model_freeze(VGG16.load_vgg16())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # video status
        self.original_fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        self.current_fps = self.original_fps
        self.roi_interval = roi_interval
        self.roi_fps = self.original_fps / self.roi_interval

        # parameters
        self.frame_counter = 0
        self.cycle_counter = 1
        self.cycle_cool_down = self.original_fps * 5
        self.cycle_sustain = 0
        self.inference_sustain = 0
        self.abnormal_count = 0
        self.continuous_abnormal_count = 0
        self.roi_cool_down = self.original_fps
        self.key = 30
        self.label_list_length = 0
        self.cos_similarity_mean = 0
        self.abnormal_compare = 99

        # state in progress
        self.cycle = False
        self.abnormal = False
        self.inference = False
        self.inference_sustainer = False
        self.abnormal_alarm_init = False
        self.alert = False
        self.sms_alarm = False

        # queue
        self.queue_frame = []
        self.queue_roi = []
        self.queue_video_difference = []
        self.queue_difference = []
        self.queue_abnormal_video = []
        self.queue_abnormal = []
        self.queue_inference_switch = []

        # log
        self.inference_log = []
        self.abnormal_log = []

    def get_answer_vectors(
        self, answer_vector_file: str = "answer_vectors_2.pkl"
    ) -> torch.Tensor:
        """정답 벡터를 호출하는 함수입니다.
        :param answer_vector_file: 파일명을 입력합니다.
        :return: answer_vectors -> 정답 벡터를 반환합니다.
        """
        with open(self.__paths.get_pickle_folder(answer_vector_file), "rb") as f:
            answer_vector = pickle.load(f)

        # answer_vectors = torch.stack(answer_vectors_list).view(
        #     [len(answer_vectors_list), 1000]
        # )
        return answer_vector

    def get_video_path(self, video: Union[str, int, None] = None):
        """비디오 경로를 호출하는 함수입니다.
        비디오를 체크하여 없을 경우 캠이 켜집니다.
        :param video: 비디오 경로를 입력받는 변수입니다.
        :return: to_play -> 비디오 파일 혹은 0(캠)을 반환합니다.
        """
        if os.path.isfile(video):
            to_play = video
        else:
            to_play = 0
        self.video_file = to_play
        return to_play

    def get_aspect_ratio(self) -> Union[list, tuple]:
        """원본 영상의 화면 비율을 반환하는 함수 입니다.
        원본 영상의 너비, 높이를 측정하여 최대 공약수로 나누어 화면 비율을 리스트로 반환합니다.
        :return: aspect_ratio -> 화면 비율
        """
        video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        gcd = math.gcd(video_width, video_height)  # 최대 공약수
        aspect_ratio = (int(video_width / gcd), int(video_height / gcd))
        return aspect_ratio

    def resize_frame(self, frame: np.ndarray, size: Union[list, tuple] = None):
        """원본 화면 비율에 맞춰 영상 크기 조절하는 함수입니다.

        aspect_ratio를 통해 원본영상의 화면 비율을 받아옵니다.
        해당 비율에 따라 영상의 크기를 자동 조절합니다.
        size에 원하는 화면 크기를 입력할 경우, 화면 비율에 상관없이 조절이 가능합니다.
        :param frame: 원본 프레임
        :param size: 원본 화면 비율에 상관없이 원하는 크기로 조절
        :return: resized_frame -> 크기 조절 된 프레임
        """
        if self.aspect_ratio == [4, 3]:
            resized_frame = cv2.resize(frame, (1024, 768))
        elif self.aspect_ratio == [16, 9]:
            resized_frame = cv2.resize(frame, (1280, 720))
        else:
            resized_frame = cv2.resize(frame, (1280, 720))

        if size is not None:
            resized_frame = cv2.resize(resized_frame, dsize=size)
        return resized_frame

    @staticmethod
    def calculate_current_fps(time_start, time_end) -> Union[float, int]:
        """
        영상 현재 프레임을 계산, 반환하는 함수입니다.
        :param time_start: 프레임 시작 부분 시간
        :param time_end: 프레임 끝 부분 시간
        :return: current_fps -> 현재 영상의 fps
        """
        spf = time_end - time_start  # 1 / frame , seconds_per_frame
        current_fps = 1 / spf
        return current_fps

    def cycle_count_reset(self):
        """
        cycle 횟수를 초기화하는 함수입니다.
        """
        self.cycle_counter = 1

    @staticmethod
    def set_ROI(
        image: np.ndarray,
        roi_x: Union[tuple, list] = (0, 100),
        roi_y: Union[tuple, list] = (0, 100),
        rectangle: bool = False,
        return_gray: bool = False,
    ) -> np.array:
        """ROI를 '비율'로 설정하는 함수입니다.

        :param image: 원본 이미지입니다. np.array의 형태이므로 이를 slice합니다.
        :param roi_x: x축의 roi 비율 0~100 설정합니다. 튜플이나 리스트로 요소쌍을 갖습니다.
        :param roi_y: y축의 roi 비율 0~100 설정합니다. 튜플이나 리스트로 요소쌍을 갖습니다.
        :param rectangle: 원본 이미지에 roi 부분을 사각형 표시합니다.
        :param return_gray: True일 경우 그레이 스케일로 반환합니다.
        :return: frame_roi -> 설정된 ROI 프레임을 반환합니다.
        """
        if len(image.shape) == 3:  # Input image 판별 (컬러 or 그레이스케일)
            image_height, image_width, image_colour_channel = image.shape  # 컬러입니다.
        elif len(image.shape) == 2:
            image_height, image_width = image.shape  # 흑백 이미지 입니다.
        else:
            raise Exception("Check input image")

        roi_x_start, roi_x_end = roi_x
        roi_y_start, roi_y_end = roi_y
        convert_x_start = round(image_width * (roi_x_start / 100))
        convert_x_end = round(image_width * (roi_x_end / 100))
        convert_y_start = round(image_height * (roi_y_start / 100))
        convert_y_end = round(image_height * (roi_y_end / 100))
        image_roi = image.copy()
        image_roi = image_roi[
            convert_y_start:convert_y_end, convert_x_start:convert_x_end
        ]
        if return_gray:
            image_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)

        if rectangle:
            cv2.rectangle(
                image,
                pt1=(convert_x_start, convert_y_start),
                pt2=(convert_x_end, convert_y_end),
                color=(0, 0, 255),
                thickness=2,
            )
        return image_roi

    def queue_frame_roi_enqueue(self, frame_roi: np.ndarray):
        """관심영역 프레임을 큐에 집어 넣는 함수입니다.

        현재 프레임 / roi_interval = 0이 되면 queue_roi에 집어넣습니다.
        :param frame_roi:
        """
        if not self.cycle:
            if self.frame_counter % self.roi_interval == 0:
                self.queue_roi.append(frame_roi)
        elif self.cycle:
            self.queue_roi.append(frame_roi)
        else:
            raise Exception("frame_roi enqueue error")

    def calculate_similarity(self, vector: torch.Tensor) -> torch.Tensor:
        """코사인 유사도를 계산해 코사인 유사도 평균을 반환합니다.

        입력 행벡터와 정답 행벡터군의 코사인 유사도를 계산합니다.
        입력 벡터하나와 정답 벡터군 n개 이므로 (n, ) 같은 열벡터의 형태로 결과값으로 나옵니다.
        결과값 내부 요소들을 평균을 취해 크기 ()의 값을 반환합니다.
        :param vector: 입력 행벡터입니다. VGG16를 Inference 하여 (1, 1000)의 크기를 가집니다.
        :return: cos_similarity_mean -> 코사인 유사도의 평균값입니다.
        """
        cos = torch.nn.CosineSimilarity()
        cos_similarity = cos(
            vector, self.answer_vectors
        )  # self.answer_vectors = 정답 벡터군
        # cos_similarity_mean = torch.mean(cos_similarity)
        return cos_similarity

    @staticmethod
    def calculate_difference_threshold(
        image1: np.ndarray, image2: np.ndarray, binary_threshold: int = 50
    ) -> np.ndarray:
        """
        2장의 이미지를 차분하여 binary_threshold 이상의 차이만 255로 할당하는 함수입니다.
        :param image1: 비교 이미지1
        :param image2: 비교 이미지2
        :param binary_threshold:
        :return: difference -> 이진화한 프레임을 반환합니다.
        """
        absolute_difference = cv2.absdiff(image1, image2)
        _, difference = cv2.threshold(
            absolute_difference, binary_threshold, 255, cv2.THRESH_BINARY
        )
        return difference

    def queue_video_difference_enqueue(
        self, binary_threshold: int, queue_length: int
    ):
        """이진화 프레임을 계산하여 queue_video_difference에 enqueue합니다.

        우선 이진화 프레임을 계산하기 위하여 사이클인 경우와 사이클이 아닌 경우를 나누어 생각합니다.
        사이클이 아닌 경우에는 정적배경 차분을 이용하기 위해 관심영역 이미지를 담은 queue를 stack 방식인
        Last In First Out 방식을 사용하여 프레임 2장을 꺼냅니다.
        사이클인 경우에는 First In First Out 방식을 사용하여 프레임 2장을 꺼냅니다.

        꺼낸 2장의 프레임을 이진화 프레임을 만들어주는 계산을 시행합니다.
        그리고 만든 이진화 프레임을 queue_video_difference에 enqueue합니다.

        :param binary_threshold: 이진화 프레임 만드는 계산을 위한 변수입니다.
         두 이미지 차분의 절대값 차이가 threshold 이상 났을때 255를 할당합니다.
        :param queue_length: queue_roi(관심영역 큐)에서 이미지를 꺼내기 시작하기 위한 queue의 길이를 설정하는 변수입니다.
        """
        if len(self.queue_roi) >= queue_length:
            if not self.cycle:
                image1 = self.queue_roi[0]
                image2 = self.queue_roi.pop(1)
            elif self.cycle:
                image1 = self.queue_roi.pop(0)
                image2 = self.queue_roi[0]
            else:
                raise Exception("queue roi error")

            binary_difference = self.calculate_difference_threshold(
                image1, image2, binary_threshold=binary_threshold
            )
            self.queue_video_difference.append(binary_difference)

    def queue_difference_enqueue(
        self, binary_difference: np.ndarray, threshold: Union[int, float] = 0.5
    ):
        """차분하여 이진화 한 프레임(binary_difference)를  queue에 담는(enqueue) 함수입니다.

        사이클이 아닌경우 사이클 구분에 사용하는 이진화 한 프레임을 enqueue합니다.
        사이클이 아닌경우 사이클 구분에 사용하는 이진화 한 프레임을 enqueue합니다.
        만약 사이클 구분에 사용하여 사이클=True 중에 필요없을 경우엔 queue를 clear합니다.

        enqueue 과정은 이진화 프레임중 nonzero인 부분의 갯수를 파악합니다.
        다음으로 프레임 사이즈 * threshold 를 곱한것보다 크면 큐에 1을 담습니다.
        작으면 0을 담습니다.

        :param binary_difference: 차분 이진화 프레임입니다.
        :param threshold: 판정을 할 때, 0이 아닌 부분을 판정하는 판정 기준입니다.
        binary_difference 의 size 에 비율을 곱합니다.
        """
        if not self.cycle:
            nonzero = np.count_nonzero(binary_difference)
            if nonzero >= binary_difference.size * threshold:
                self.queue_difference.append(1)
            elif nonzero < binary_difference.size * threshold:
                self.queue_difference.append(0)
        elif self.cycle:
            if len(self.queue_difference) != 0:
                self.queue_difference = []
        else:
            raise Exception("ROI calculate error")

    def queue_difference_dequeue(self, queue_length_sec: Union[int, float] = 0.5):
        """queue_difference 를 dequeue 하는 함수입니다.

        :param queue_length_sec: queue 의 길이를 정하는 변수입니다. 몇 초 동안을 queue 에 담을건지 정합니다.
        """
        if len(self.queue_difference) >= self.original_fps * queue_length_sec:
            del self.queue_difference[0]

    def renew_roi_background(
        self, frame_roi: np.array, cool_down: Union[int, float] = 3
    ):
        """cool down 마다 정적배경차분의 배경을 갱신하는 함수입니다.

        사이클이 아닌경우, 3초마다 정적배경차분의 기준배경을 갱신합니다.
        :param frame_roi: 관심영역 프레임 입니다.
        :param cool_down: 기준배경 갱신시간 입니다. 기본값은 3초입니다.
        """
        if not self.cycle:
            if self.roi_cool_down <= 0:
                self.queue_roi[0] = frame_roi
                self.roi_cool_down = round(self.original_fps * cool_down)
        elif self.cycle:
            pass
        else:
            raise Exception("renew error")
        self.roi_cool_down -= 1

    def cycle_judge(self, cycle_cool_down: Union[int, float], difference_threshold: Union[int, float]):
        """ 사이클을 판정하는 함수입니다.

        cycle_cool_down 이 0 이하이면 queue_difference 에 있는 판정을 측정합니다.
        판정을 측정하여 1이 difference_threshold 이상이면 cycle 판정을 내리고 alarm 을 냅니다.
        그리고 다시 설정된 cool_down 만큼 시간이 설정이 되며 1프레임 마다 1만큼의 cool_down 이 줄어듭니다.
        cool_down = fps(frame per second) * cycle_cool_down 변수
        :param cycle_cool_down: 쿨다운의 시간입니다. 단위는 초입니다.
        :param difference_threshold: queue_difference 에서 판정을 측정할때의 비율입니다.
        """

        if self.cycle_cool_down <= 0:
            if (
                self.queue_difference.count(1)
                >= len(self.queue_difference) * difference_threshold
            ):
                self.cycle = True
                self.cycle_alarm()
                self.cycle_cool_down = round(self.original_fps * cycle_cool_down)
        self.cycle_cool_down -= 1

    def queue_video_difference_dequeue(
        self, frame_roi: np.ndarray, queue_length: int
    ) -> np.ndarray:
        """

        :param frame_roi:
        :param queue_length:
        :return:
        """
        if len(self.queue_video_difference) >= queue_length:
            binary_difference = self.queue_video_difference.pop(0)
        else:
            binary_difference = np.zeros_like(frame_roi)
        return binary_difference

    def cycle_sustain_set(self, cycle_sustain_time: Union[int, float] = 2):
        """사이클 판정을 유지할 시간을 설정하는 함수입니다.

        사이클인 경우 cycle_sustain 이 1씩 증가됩니다.
        증가된 cycle_sustain 이 설정해둔 time 보다 길어지게 되면 cycle을 off 합니다.
        :param cycle_sustain_time: 사이클을 유지하는 시간의 변수입니다. 단위는 초입니다.
        """
        if self.cycle:
            self.cycle_sustain += 1
        elif not self.cycle:
            self.cycle_sustain = 0
        else:
            raise Exception("cycle sustain error")

        if (
            self.cycle_sustain > self.original_fps * cycle_sustain_time
        ):  # cycle sustain time 초 동안 활성화 후 비활성화
            self.cycle = False

    def inference_switch(
        self,
        binary_difference: np.ndarray,
        n_inference_frame: int = 8,
        switch_threshold: Union[int, float] = 0.05,
    ):
        """inference 를 시행할지 말지 결정하는 함수입니다.

        cycle 단일 판정으로는 균일한 inference 타이밍을 잡기가 힘들어 추가한 장치입니다.
        도포하는 장치가 양끝 중, 한곳으로 붙어 움직임이 없을 때 inference 의 스위치가 True 가 됩니다.
        :param binary_difference: 이진화 차분의 입력 데이터입니다.
        :param n_inference_frame: inference 를 몇 프레임 시행할것인지 결정하는 변수입니다.
        :param switch_threshold: 스위치의 임계값입니다.
        """
        if self.cycle:
            if (
                np.count_nonzero(binary_difference)
                < binary_difference.size * switch_threshold
            ):
                self.queue_inference_switch.append(1)
            else:
                self.queue_inference_switch.append(0)

            if len(self.queue_inference_switch) > self.original_fps * 0.3:
                del self.queue_inference_switch[0]

                if (
                    np.count_nonzero(self.queue_inference_switch)
                    >= len(self.queue_inference_switch) / 2
                ):
                    if not self.inference_sustainer:
                        self.inference = True
                        self.inference_sustainer = True
        else:
            self.queue_inference_switch = []
            self.inference_sustainer = False

        if self.inference:
            self.inference_sustain += 1

        if self.inference_sustain > n_inference_frame:
            self.inference = False
            self.inference_sustain = 0

        self.abnormal_compare = n_inference_frame

    def extract_frame(self, frame: np.ndarray):
        """프레임 캡쳐 추출하는 함수
        :param frame:
        :return:
        """
        path = GetProjectPath().get_data_folder("new_normal")
        os.makedirs(path, exist_ok=True)
        contemp = uuid.uuid1()
        cv2.imwrite(os.path.join(path, f"{contemp.hex}.png"), frame)

    def on_cycle_inference(self, frame_resize, inference_start_cycle, extract=False):
        """모델에 Inference 를 진행해, vector 를 추출하는 함수입니다.

        모델에 inference 하여 벡터를 추출하고 그 벡터를 정답 벡터군과 코사인 유사도를 구해 전역변수에 할당합니다.
        :param frame_resize: 입력 프레임
        :param inference_start_cycle: 몇번째 사이클부터 inference 를 진행할 것인지 정하는 변수
        :param extract:
        """
        if self.inference and self.cycle_counter - 1 >= inference_start_cycle:
            self.abnormal_alarm_init = True
            # print(f'inference: {self.inference_sustain}')
            ROI = self.set_ROI(frame_resize, (25, 75), (50, 100))
            inference_frame = self.inference_preprocess(ROI)

            if extract:
                self.extract_frame(ROI)

            output = self.model(inference_frame)

            # cos similarity
            cos_similarity = self.calculate_similarity(output)
            self.cos_similarity_mean += cos_similarity

    def cycle_alarm(self):
        """
        사이클을 표시하는 함수입니다.
        """
        print(f"==== Cycle {self.cycle_counter} ====")
        self.cycle_counter += 1

    def abnormal_judge(self, abnormal_threshold: Union[float, int]):
        """이상을 판정하는 함수입니다.

        :param abnormal_threshold:
        """
        if self.inference_sustain <= self.abnormal_compare:
            if self.cos_similarity_mean / self.abnormal_compare < abnormal_threshold:
                self.abnormal = True
            else:
                self.abnormal = False

    def abnormal_alarm(self, patient=3):
        """이상판정에 대한 알림을 주는 함수입니다.
        정상일때와 비정상일때 각각의 알람을 표시하는 함수입니다.
        """
        if self.inference_sustain == self.abnormal_compare:
            if self.abnormal_alarm_init:
                print(
                    f"Normal Similarity  : {self.cos_similarity_mean / self.abnormal_compare}"
                )
                self.inference_log.append(
                    self.cos_similarity_mean / self.abnormal_compare
                )

                if self.abnormal:  # 이상일 경우
                    self.abnormal_count += 1  # 이상 카운트 횟수 +1
                    self.continuous_abnormal_count += 1  # 연속 이상 카운트 + 1
                    print(f"Abnormal Occurred {self.abnormal_count}")
                    self.abnormal_log.append(self.cycle_counter - 1)
                    self.alert = True
                else:
                    self.continuous_abnormal_count = 0
                    print(f"Normal")
                    self.alert = False
                self.cos_similarity_mean = 0
                self.continuous_abnormal_alarm(patient=patient)
            else:
                print("Cycle reached yet.")

    def continuous_abnormal_alarm(self, patient: int = 3):
        """연속으로 abnormal 이 판정될 경우 알람을 띄우는 함수입니다.

        :param patient: 알람을 띄우기 위해 연속으로 떠야하는 abnormal 판정 회수입니다.
        :return:
        """
        if self.continuous_abnormal_count >= patient:
            self.sms_alarm = True
            print(
                f"Abnormal occurred continuously {patient} times. Please fix the process."
            )
        else:
            print(
                f"Continuous Abnormal Occurred [{self.continuous_abnormal_count}/{patient}] times"
            )

    def inference_preprocess(self, img: np.ndarray) -> torch.Tensor:
        """
        inference 할 프레임을 전처리 하는 함수입니다.
        :param img:
        :return:
        """
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, [224, 224])
        image = np.transpose(image, axes=[2, 0, 1])
        image = torch.tensor(image).float().to(self.device) / 255
        image = torch.unsqueeze(image, 0)
        return image

    def control_waitkey(self, aim_fps: Union[float, int]):
        """
        녹화 영상을 플레이 할 경우, waitkey를 조절하여 자동으로 영상속도를 맞추는 함수입니다.
        :param aim_fps: 목표 영상속도 fps
        """
        if self.current_fps > aim_fps:
            self.key = self.key + 1
        elif self.current_fps < aim_fps:
            self.key = self.key - 1
        else:
            self.key = self.key

        # key가 너무 빠른 경우 reset
        if self.key < 2:
            self.key = 1


if __name__ == "__main__":
    pass
