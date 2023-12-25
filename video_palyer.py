import cv2


class VideoPlayer:
    def __init__(self, video_path=None, buffer_length=10):
        self.cap = self.prepare_video(video_path)
        self.play_buffer = []
        self.ret_buffer = []
        self.buffer_length = buffer_length

    def is_buffer_full(self):
        if len(self.play_buffer) >= self.buffer_length:
            return True
        else:
            return False
        
    def get_buffer(self) -> list:
        return self.play_buffer
    
    def prepare_video(self, video_path):
        if video_path is None:
            return cv2.VideoCapture(0)
        else:
            return cv2.VideoCapture(video_path)

    def fill_buffer(self, frame):
        if self.is_buffer_full():
            del self.play_buffer[0]
        self.play_buffer.append(frame)

    def get_single_frame(
        self,
    ):
        ret, frame = self.cap.read()
        if not ret:
            return None
        else:
            return frame

    def play(self, title=None):
        while True:
            ret, frame = self.cap.read()

            # Get frame from buffer
            if self.is_buffer_full():
                frame = self.play_buffer.pop(0)
                ret = self.ret_buffer.pop(0)
            else:
                self.play_buffer.append(frame)
                self.ret_buffer.append(ret)
                continue

            # Check Frame exist
            if not ret:
                break

            # Title
            if title is None:
                title = "Video"

            cv2.imshow(title, frame)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

        self.cap.release()


if __name__ == "__main__":
    player = VideoPlayer(video_path="/Users/kangnam/dataset/metal_printer.avi")
    player.play()
