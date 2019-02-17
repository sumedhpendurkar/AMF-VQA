#Source Code https://github.com/asharma327/Read_Gif_OpenCV_Python
#Modified version of the original repo
import cv2
import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
def ConvertVideoToFrames(vid_path):
    video = cv2.VideoCapture(vid_path)
    frame_num = 0
    frame_list = []
    while True:
        try:
            okay, frame = video.read()
            frame_list.append(frame)
            if not okay:
                break
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break
    return frame_list

def OutputFramesAsPics(frame_list):
    frame_list_reduce = frame_list[0::2]
    path = os.getcwd()
    folder_name = 'Vid_to_Images'
    if not os.path.exists(path + '/' + folder_name):
        os.makedirs(path + '/' + folder_name)
    for frames_idx in range(len(frame_list_reduce)):
        cv2.imwrite(os.path.join(path + '/' + folder_name, str(frames_idx+1) + '.jpeg'), frame_list_reduce[frames_idx])

    pass
if __name__=="__main__":
    frames = ConvertVideoToFrames('/home/sameer/Desktop/Stuff/BTech Project/test.avi')
    resize = transforms.Compose([transforms.ToPILImage(), transforms.Resize((200,200)), transforms.ToTensor()])
    timeDepth = len(frames)
    print(timeDepth)
    #Error in processing last frame leads to NoneType Errors
    t_frames = torch.FloatTensor(3, timeDepth, 200, 200)
    for f in range(timeDepth):
        frame = torch.from_numpy(frames[f])
        frame = resize(frame)
        t_frames[:, f, :, :] = frame
    print(t_frames)
