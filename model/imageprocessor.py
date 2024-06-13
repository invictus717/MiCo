import os
import random
from PIL import Image

import torch
from torchvision.transforms.transforms import *
from torchvision import transforms


class ImageProcessor(object):
    def __init__(self, image_resolution, image_encoder_type, image_transforms='none', training=True):
        self.training = training
        
        self.resolution = image_resolution
        self.image_encoder_type = image_encoder_type

        if image_encoder_type.startswith('clip') or image_encoder_type.startswith('evaclip'):
            self.mean = [0.48145466, 0.4578275, 0.40821073] 
            self.std  = [0.26862954, 0.26130258, 0.27577711]
        else:       
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]   
        
        self.image_transforms = image_transforms
        if image_transforms == 'none':
            self.train_transforms = transforms.Compose([Resize((self.resolution,self.resolution)),
                                                        Normalize(self.mean,self.std)])
                
            self.test_transforms = transforms.Compose([Resize((self.resolution,self.resolution)),
                                                        Normalize(self.mean,self.std)])
        elif image_transforms == 'crop_flip':
            self.train_transforms = transforms.Compose([RandomResizedCrop(self.resolution, [0.8,1.0],[1.0,1.0]),
                                                        RandomHorizontalFlip(),
                                                        Normalize(self.mean,self.std)])

            self.test_transforms = transforms.Compose([Resize(self.resolution),
                                    CenterCrop(self.resolution),
                                    Normalize(self.mean,self.std)])                           
        else:
            raise NotImplementedError 
            
    def __call__(self, image_file):

        try:
            img_path = image_file
            if not os.path.exists(img_path):
                print('not have image', image_file)
                return None
            img = Image.open(img_path)
            img = img.convert('RGB')  #### convert 1-channel gray image and 4-channel CMYK image to RGB image
            img = transforms.ToTensor()(img)
            if self.training:    
                img = self.train_transforms(img)
            else:
                img = self.test_transforms(img)

            img = img.unsqueeze(0)

            return img

        except Exception as e:
            print(e)
            return None

if __name__ == "__main__":
    image_file = "./data/test.jpeg"
    proc = ImageProcessor(image_resolution=224, image_encoder_type="swin", training=True)
    image_input = proc(image_file)
    print(image_input.size())