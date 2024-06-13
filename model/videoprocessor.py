import os
import random
from decord import VideoReader
from PIL import Image

import torch
from torchvision.transforms.transforms import *
from torchvision import transforms


def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num:   ###padding with the last frame
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]

class VideoProcessor(object):
    def __init__(self, video_resolution, video_encoder_type, sample_num = 4, video_transforms='none', data_format="frame", training=True):
        self.frame_syncaug = True
        self.training = training
        self.sample_num = sample_num 
        self.data_format = data_format
        
        self.resolution = video_resolution
        self.video_encoder_type = video_encoder_type

        if video_encoder_type.startswith('clip') or video_encoder_type.startswith('evaclip'):
            self.mean = [0.48145466, 0.4578275, 0.40821073] 
            self.std  = [0.26862954, 0.26130258, 0.27577711]
        else:       
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]   
        
        self.video_transforms = video_transforms
        if video_transforms == 'none':
            self.train_transforms = transforms.Compose([Resize((self.resolution,self.resolution)),
                                                        Normalize(self.mean,self.std)])
                
            self.test_transforms = transforms.Compose([Resize((self.resolution,self.resolution)),
                                                        Normalize(self.mean,self.std)])
        elif video_transforms == 'crop_flip':
            self.train_transforms = transforms.Compose([RandomResizedCrop(self.resolution, [0.8,1.0],[1.0,1.0]),
                                                        RandomHorizontalFlip(),
                                                        Normalize(self.mean,self.std)])

            self.test_transforms = transforms.Compose([Resize(self.resolution),
                                    CenterCrop(self.resolution),
                                    Normalize(self.mean,self.std)])                           
        else:
            raise NotImplementedError 
            
    def __call__(self, video_file):

        video_pixels = []        
        sample_num = self.sample_num
        try:
            if self.data_format == 'frame':
                frame_path = video_file
                if not os.path.exists(video_path):
                    print('not have videos', video_file)
                    return None
                frames = os.listdir(frame_path)
                frames.sort()   ### ['img_0001.jpg','img_0002.jpg',...]
                sample_num = self.sample_num
                frames_splited = split(frames,sample_num)    
                if self.training:
                    sample_idx = [random.choice(i) for i in frames_splited]
                else:
                    sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited]
                for i in range(sample_num):
                    frame = Image.open(os.path.join(frame_path,sample_idx[i]))
                    frame = transforms.ToTensor()(frame)   ## frame: 3XhXw
                    video_pixels.append(frame.unsqueeze(0))

            elif self.data_format == 'raw':
                video_path = video_file
                if not os.path.exists(video_path):
                    print('not have videos', video_file)
                    return None
                container = decord.VideoReader(uri=video_path)    
                frames_ids = list(range(len(container)))
        
                frames_splited = split(frames_ids, sample_num)
                if self.training:
                    sample_idx = [random.choice(i) for i in frames_splited]
                else:
                    sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited] 

                frames = container.get_batch(sample_idx).asnumpy()
                # print(len(frames), type(frames),sample_idx)

                for i in frames: 
                    frame = Image.fromarray(i)
                    frame = transforms.ToTensor()(frame)   ## frame: 3XhXw
                    video_pixels.append(frame.unsqueeze(0))


            video_pixels = torch.cat(video_pixels,dim=0)   ### nX3xHxW
            if self.training:
                video_pixels = self.train_transforms(video_pixels)    
            else:
                video_pixels = self.test_transforms(video_pixels)     
            return video_pixels

        except Exception as e:
            print(e)
            print(video_file)
            return None

if __name__ == "__main__":
    video_file = "./data/test.mp4"
    proc = VideoProcessor(video_resolution=224, video_encoder_type="swin", sample_num=4, data_format="raw", training=True)
    video_input = proc(video_file)
    print(video_input.size())