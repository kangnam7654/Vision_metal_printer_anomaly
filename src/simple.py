import torch
import cv2
import argparse

from utils.data_process.inference_data_processing import image_preprocess
from lib.models.resnet_backbone import get_res_back    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", default=None, type=str)
    
    args = parser.parse_args()
    
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cuda':
        device = 'cuda'
    elif args.device == 'cpu':
        device = 'cpu'
    else:
        raise KeyError("Set device arg among ['cuda', 'cpu', None]")
    
    video_path = ''
    cap = cv2.VideoCapture(video_path)
    
    model = get_res_back()
    model = model.cuda()
    while True:
        ret, frame = cap.read()
        if ret:
            image = image_preprocess(frame)
            output = model(image)
            print(output)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        else:
            print('error')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()