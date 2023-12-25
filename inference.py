import argparse
import torch
import numpy as np
import cv2
from video_palyer import VideoPlayer
from models.encoder import ResNetSimCLR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path", type=str, default="/Users/kangnam/dataset/metal_printer.avi"
    )
    parser.add_argument("--buffer_length", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    return args


def preprocess(buffer, device):
    temp = []
    for image in buffer:
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        temp.append(image)
    image = np.stack(temp, axis=0)
    image = torch.from_numpy(image)
    image = torch.permute(image, (0, 3, 1, 2)).float().to(device)
    image = image / 255
    return image


def main(args):
    player = VideoPlayer(args.video_path, args.buffer_length)
    model = ResNetSimCLR(base_model="resnet18")
    model.eval()
    model.to(args.device)

    alert = False

    while True:
        suspicious = 0
        frame = player.get_single_frame()
        player.fill_buffer(frame)

        if player.is_buffer_full():
            cv2.imshow("video", frame)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()

            buffer = player.get_buffer()
            data = preprocess(buffer, args.device)
            outs, _ = model(data)
            mean_outs = torch.mean(outs, dim=0)
            for out in outs:
                sim = torch.cosine_similarity(mean_outs, out, dim=0)

                if sim <= args.threshold:
                    suspicious += 1
            if suspicious >= 5:
                alert = True

            if alert:
                print("Anomaly")


if __name__ == "__main__":
    args = get_args()
    main(args)
