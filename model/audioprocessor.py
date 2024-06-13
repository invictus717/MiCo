import os
import random

import torch
import torchaudio


def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num:   ###padding with the last frame
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]


class AudioProcessor(object):
    def __init__(self, melbins, target_length, sample_num, frame_shift=10, resize_melbin_num=224, mean=15.41663, std=6.55582, training=True):
        self.melbins = melbins
        self.target_length = target_length
        self.training = training
        self.frame_shift = frame_shift
        self.sample_num = sample_num
        self.resize_melbin_num = resize_melbin_num
        
        self.mean = mean
        self.std = std

    def __call__(self, wav_file):

        if not os.path.exists(wav_file):
            print('not have audios', wav_file)
            return torch.zeros(self.sample_num, self.target_length, self.melbins)

        try:
            waveform, sr = torchaudio.load(wav_file)
            if sr != 16000:
                trans = torchaudio.transforms.Resample(sr, 16000)
                waveform = trans(waveform)
        
            waveform = waveform * 2 ** 15
            fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=self.melbins, sample_frequency=16000, frame_length=25, frame_shift=10)

            if fbank.size(1) != self.resize_melbin_num:
                fbank = torch.nn.functional.interpolate(fbank.reshape(1, 1, *fbank.shape[-2:]), size=(fbank.size(0), self.resize_melbin_num), mode='bilinear').reshape(fbank.size(0), self.resize_melbin_num)

            # ### normalization
            fbank = (fbank - self.mean) / (self.std * 2)
            src_length = fbank.shape[0]
            # #### sample 

            output_slices = []
            pad_len = max(self.target_length * self.sample_num -src_length, self.target_length - src_length%self.target_length)
            fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_len))(fbank)

            total_slice_num = fbank.shape[0] // self.target_length
            total_slice_num = list(range(total_slice_num))
            total_slice_num = split(total_slice_num, self.sample_num)
            
            if self.training:
                sample_idx = [random.choice(i) for i in total_slice_num]
            else:
                sample_idx = [i[(len(i)+1)//2-1] for i in total_slice_num]

            
            for i in sample_idx:
                cur_bank = fbank[i*self.target_length : (i+1)*self.target_length]
                output_slices.append(cur_bank)

                    

            fbank = torch.stack(output_slices,dim=0)   ### n, 1024, 128

            
            return fbank
        

        except Exception as e:
            print(e)
            return    


if __name__ == "__main__":
    wav_file = "./data/test.flac"
    proc = AudioProcessor(melbins=224, target_length=224, sample_num=4, training=True)
    audio_input = proc(wav_file)
    print(audio_input.size())