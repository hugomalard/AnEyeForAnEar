import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys
import torch
import torchaudio
import numpy as np
import os
os.chdir('./LLaVA')
import argparse
import torch
import json
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image
import torch.nn as nn
import requests
from PIL import Image
from io import BytesIO
import re
from transformers import AutoProcessor
import torchvision
from transformers import AutoProcessor,ClapModel
sys.path.append('../cav-mae/src')
from models import CAVMAE
import json

parser = argparse.ArgumentParser()
parser.add_argument("--frame_path", type=str, default="")
parser.add_argument("--audio_path", type=str, default="")
parser.add_argument("--model-path", type=str, default="DALI_OT_ATT.pt")
parser.add_argument("--original-model", type=str, default="facebook/opt-350m")
parser.add_argument("--save-path", type=str, default="pseudoLab.json")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--sep", type=str, default=",")
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=512)
args = parser.parse_args()


trans = torchvision.transforms.ToPILImage()
trans1 = torchvision.transforms.ToTensor()
def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def wav2fbank(filename, filename2=None, mix_lambda=-1,beats = False):
    # no mixup
    if filename2 == None:
        waveform, sr = torchaudio.load(filename)
        resampler = torchaudio.transforms.Resample(sr,16000)
        waveform = resampler(waveform)
        if waveform.shape[0]>1:
            waveform = waveform[0,:].unsqueeze(0)

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
        if waveform.shape[1]>0:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        else:
            with open('./ACTestBug.txt', 'a') as f:
                f.write('\n'+str(filename))
            fbank = torch.zeros([512, 128]) + 0.01
    except:
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

        fbank = torch.zeros([512, 128]) + 0.01
        print('there is a loading error')



    target_length = 1024
    n_frames = fbank.shape[0]

    p = target_length - n_frames

    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    return fbank
    

def load_mel(audio_file,audio_processor,cav=True):
    if not cav:
        audio,sr = torchaudio.load(audio_file)
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr,48000)
            audio = resampler(audio)
            sr = 48000
        if audio.shape[0] > 1:
            audio = audio_processor(audios=np.array(audio[0]), return_tensors="pt", padding=True,sampling_rate=sr)['input_features']
        else:
            audio = audio_processor(audios=np.array(audio), return_tensors="pt", padding=True,sampling_rate=sr)['input_features']
    else:
        fbank = wav2fbank(audio_file)
        audio = (fbank - (-5.081)) / (4.4849) 
        audio = audio.unsqueeze(0).to(dtype=torch.bfloat16)
    return audio



def load_audios(audio_files,processor):
    out = []
    for audio_file in audio_files:
        audios = load_mel(audio_file,processor)
        out.append(audios)
    return torch.stack([o[0] for o in out])

class FilePathDataset(Dataset):
    def __init__(self, dataframe,audio_processor, audio_path_column='file_path',image_path_column='img_path',return_img=True,return_audio=True):
        self.dataframe = dataframe
        self.audio_processor = audio_processor
        self.audio_path_column = audio_path_column
        self.image_path_column = image_path_column
        self.return_audio = return_audio
        self.return_image = return_img

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):
        idd=self.dataframe.iloc[idx]['file_id']

        if self.return_audio:
            filename= args.audio_path+self.dataframe.iloc[idx]['file_id']+'.wav'
            audios =load_mel(filename,self.audio_processor)
        else:
            audios = torch.normal(0,1,(2,1000,64))
        if self.return_image:
            path = args.frame_path+'frame_5/' +self.dataframe.iloc[idx]['file_id']+'.jpg'
            images = trans1(load_image(path)) 
        return audios,images,'',idd


disable_torch_init()



model_base ='liuhaotian/llava-v1.5-7b'
model_name = get_model_name_from_path(model_base)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_base, None,model_name,audio='laion/clap-htsat-unfused',modality='audio_vis',prefix=True
)
model=model.to(dtype=torch.bfloat16)


import numpy as np

audio_model = CAVMAE(audio_length=1024, #\  all models trained with 10s audio
                    modality_specific_depth=11,# \ # all models trained with 11 modality-specific layers and 1 shared layer
                    norm_pix_loss=True, tr_pos=False,use_clip=True,dali=True,x_attn=True,method='emd') # most models are trained with pixel normalization and non-trainabe positional embedding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl_weight = torch.load(args.model_path+'/audio_backbone.pth', map_location=device)
from collections import OrderedDict
new_state_dict = OrderedDict()


for k, v in mdl_weight.items(): 
    new_state_dict[k.replace("module.", "")] = v
miss, unexpected = audio_model.load_state_dict(new_state_dict, strict=False)

model.model.audio_tower = audio_model.to(dtype=torch.bfloat16,device='cuda')
model.model.audio_tower.cav=True

model.mask = 0
model.noise = 0
qs = 'Describe the sound that can be heard in this scene.'
image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
if IMAGE_PLACEHOLDER in qs:
    if model.config.mm_use_im_start_end:
        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    else:
        qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
else:
    if model.config.mm_use_im_start_end:
        qs = image_token_se + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

if "llama-2" in model_name.lower():
    conv_mode = "llava_llama_2"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"

if args.conv_mode is not None and conv_mode != args.conv_mode:
    print(
        "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
            conv_mode, args.conv_mode, args.conv_mode
        )
    )
else:
    args.conv_mode = conv_mode

conv = conv_templates[args.conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

df =pd.read_csv('/tsi/dcase/AudioSet/metadata.csv') #metadat of audioset to get the balance train
df = df[df['set']=='balanced_train']
df = df.drop_duplicates(subset='file_id')
df = df.sample(frac=1).reset_index(drop=True)

captions = []
ids = []

audio_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
file_path_dataset = FilePathDataset(df,audio_processor, audio_path_column='audios_path',image_path_column='img_path',return_img=True,return_audio=True)

batch_size = 1  # Batch size of 1 to not rescale any image, but can be resized and batched for much faster inference
file_path_dataloader = DataLoader(file_path_dataset, batch_size=batch_size, shuffle=False,num_workers=8)

print('Using prefix tuned')

prefix = torch.load(args.model_path+'/prefix.bin')
model.model.audio_tower.prefix=torch.nn.Parameter(prefix['model.audio_tower.prefix'], requires_grad=False).to(dtype=torch.bfloat16,device='cuda') #model.model.audio_tower.prefix.to(dtype=torch.bfloat16)


gen_captions = []
idsL = []
print('infering on',len(file_path_dataloader))
for i, (audios,images,captions,ids) in enumerate(file_path_dataloader):
    idsL.append(ids)
    images = [trans(images[i]) for i in range(images.shape[0])]
    audios = torch.stack([o[0] for o in audios])
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.bfloat16)


    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    inputs_ids = torch.cat([input_ids for i in range(audios.shape[0])])
    image_sizes = [x.size for x in images]
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=128,
            use_cache=True,
            audios = audios.to(dtype=torch.bfloat16,device='cuda'),
            only_im = False
        )
        out = []
        for k in range(len(output_ids)):
            out.append(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[k].strip())

    gen_captions.extend(out)
    if i %10==0 and i!=0:
        print(i)
        
idsL = [idsL[i][0] for i in range(len(idsL))]
df = pd.DataFrame({'file_id':idsL,'caption':gen_captions})
mask = df['file_id'] == 'error'
df = df[~mask]

data=[]
for i in range(len(df.index)):
    if type(df['caption'].iloc[i])==float:
        continue
    dic={}
    dic['id']=str(df.iloc[i]['file_id'])
    dic['image'] = args.frame_path+'/frame_5/'+str(df.iloc[i]['file_id'])+'.jpg'
    dic['audio'] = args.audio_path+str(df.iloc[i]['file_id'])+'.wav'
    conv = []
    conv.append({'from':'human','value':"<image>\nDescribe the sound that can be heard in this scene."})
    conv.append({'from':'gpt','value':df.iloc[i]['caption']})
    dic['conversations'] = conv
    data.append(dic)

with open(args.save_path, 'w') as f:
    json.dump(data, f)

