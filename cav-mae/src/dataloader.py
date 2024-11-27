# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader_old.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import os.path

import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
from PIL import Image
import PIL
from transformers import CLIPImageProcessor,AutoProcessor,AutoFeatureExtractor
import pandas as pd
def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf,precise_frame = False, return_id=False, label_csv=None,beats=False,use_clip=False,use_clap=False,img2img=False,ar=False):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        self.return_id = return_id
        self.precise_frame = precise_frame
        self.beats = beats
        self.img2img=img2img
        self.use_clip = use_clip
        if use_clip:
            self.image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')

        self.use_clap=use_clap
        if self.use_clap:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")

        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.data = self.pro_data(self.data)
        print('Dataset has {:d} samples'.format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.ar = ar
        self.get_id = audio_conf.get('get_id',False)
        self.label_smooth = self.audio_conf.get('label_smooth', 0.0)
        print('Using Label Smoothing: ' + str(self.label_smooth))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

        self.target_length = self.audio_conf.get('target_length')

        # train or eval
        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get('frame_use', -1)
        # by default, 10 frames are used
        self.total_frame = self.audio_conf.get('total_frame', 10)
        print('now use frame {:d} from total {:d} frames'.format(self.frame_use, self.total_frame))

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))
        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )])

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        if self.precise_frame == True:
            for i in range(len(data_json)):
                data_json[i] = [data_json[i]['wav'], data_json[i]['labels'], data_json[i]['video_id'], data_json[i]['video_path'],data_json[i]['frame']]
        else:
            for i in range(len(data_json)):
                data_json[i] = [data_json[i]['wav'], data_json[i]['labels'], data_json[i]['video_id'], data_json[i]['video_path']]            
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum['wav'] = np_data[0]
        datum['labels'] = np_data[1]
        datum['video_id'] = np_data[2]
        datum['video_path'] = np_data[3]
        if np_data.shape[0] == 5:
            datum['chosen_frame'] = np_data[4]
        return datum

    def get_image(self, filename, filename2=None, mix_lambda=1,img2img=False):
        if self.use_clip and not img2img:
            image = Image.open(filename).convert('RGB')
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            return torch.Tensor(image)
        
        if filename2 == None:
            img = Image.open(filename)
            image_tensor = self.preprocess(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1,beats = False):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            resampler = torchaudio.transforms.Resample(sr,16000)
            waveform = resampler(waveform)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            if beats == True:
                fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
                return fbank
            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')



        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def randselect_img(self, video_id, video_path,chosen_frame=None,use_clip=False):
        if self.mode == 'eval':
            # if not specified, use the middle frame
            if self.frame_use == -1:
                frame_idx = int((self.total_frame) / 2)
            else:
                frame_idx = self.frame_use
        else:
            frame_idx = random.randint(0, 9)

        if chosen_frame:
            frame_idx = chosen_frame
        #if use_clip:
        frame_idx=0
        while os.path.exists(video_path + '/frame_' + str(frame_idx) + '/' + video_id + '.jpg') == False and frame_idx >= 1:

            frame_idx -= 1
        out_path = video_path + '/frame_' + str(frame_idx) + '/' + video_id + '.jpg'
        return out_path

    def __getitem__(self, index):
        if random.random() < self.mixup:
            datum = self.data[index]
            datum = self.decode_data(datum)
            mix_sample_idx = random.randint(0, self.num_samples-1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)
            try:
                fbank = self._wav2fbank(datum['wav'], mix_datum['wav'], mix_lambda)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio')
            try:
                if self.precise_frame:
                    image = self.get_image(self.randselect_img(datum['video_id'], datum['video_path'],datum['chosen_frame']), self.randselect_img(mix_datum['video_id'], datum['video_path']), mix_lambda)
                else: 
                    image = self.get_image(self.randselect_img(datum['video_id'], datum['video_path']), self.randselect_img(mix_datum['video_id'], datum['video_path']), mix_lambda)
            except:
                image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
                print('there is an error in loading image')
                print(datum['video_id'])

            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (1.0 - self.label_smooth)
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (1.0 - self.label_smooth)
            label_indices = torch.FloatTensor(label_indices)

        else:
            missing = []
            datum = self.data[index]
            datum = self.decode_data(datum)
            # label smooth for negative samples, epsilon/label_num
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            #try:
            if self.beats == True:
                fbank = self._wav2fbank(datum['wav'], None, 0,beats=self.beats)
            elif self.use_clap:
                try:
                    audio,sr = torchaudio.load(datum['wav'])
                except:
                    audio=torch.normal(0,1,(1,480000))
                    sr=48000
                    print('error with ',datum['wav'])
                if sr != 48000:
                    resampler = torchaudio.transforms.Resample(sr,48000)
                    audio = resampler(audio)
                    sr = 48000
                if audio.shape[0] > 1:
                    audio = audio[0].unsqueeze(0)
                if audio.shape[1]<480000:
                    padding_left = 0
                    padding_right = 480000 - audio.size(1)
                    # Pad the tensor with zeros
                    audio = torch.nn.functional.pad(audio, (padding_left, padding_right), value=0)

                mel = []
                for i in range(3):
                    end = (i+8)*48000 if audio.shape[1]>=(i+8)*48000 else audio.shape[1]
                    inputs1 = self.feature_extractor(np.array(audio)[0][int(48000*i):end], return_tensors="pt",sampling_rate=48000)
                    mel.append(inputs1)
                    if end == audio.shape[1]:
                        break
                
                melF = [me['input_features'] for me in mel]

                fbank=torch.cat(melF,dim=1)
                    
   
            else:
                try:
                    fbank = self._wav2fbank(datum['wav'], None, 0)
                except:
                    fbank = torch.zeros([self.target_length, 128]) + 0.01
                    print('there is an error in loading audio')
                    print(datum['wav'])
                    missing.append(datum['wav'])
            try:
                if self.precise_frame:
                    image = self.get_image(self.randselect_img(datum['video_id'], datum['video_path'],datum['chosen_frame']), None, 0)
                else:
                    image = self.get_image(self.randselect_img(datum['video_id'], datum['video_path']), None, 0)
                    if self.img2img:
                        image2 = self.get_image(self.randselect_img(datum['video_id'], datum['video_path']), None, 0,img2img=True)
            except:
                if not self.use_clip:
                    image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
                else:
                    image = torch.zeros([3, 336, 336]) + 0.01
                
                if self.img2img:
                    image2 = torch.zeros([3, self.im_res, self.im_res]) + 0.01

                print(datum['video_id'])
                print('there is an error in loading image')

                missing.append(datum['video_path'])

            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        if not self.use_clap:
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = torch.transpose(fbank, 0, 1)
            fbank = fbank.unsqueeze(0)
            if self.freqm != 0:
                fbank = freqm(fbank)
            if self.timem != 0:
                fbank = timem(fbank)
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)

            # normalize the input for both training and test
            if self.skip_norm == False:
                fbank = (fbank - self.norm_mean) / (self.norm_std)
            # skip normalization the input ONLY when you are trying to get the normalization stats.
            else:
                pass

            if self.noise == True:
                fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
                fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)
        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        if self.return_id:
            return fbank, image,index

        return fbank, image, label_indices

    def __len__(self):
        return self.num_samples