import json
import os
import torch,timm
import torch.nn as nn
from torch.cuda.amp import autocast
from numpy import dot
from numpy.linalg import norm
import numpy as np
assert timm.__version__ == '0.4.5' # it is important to have right version of timm
os.chdir('/home/ids/hmalard/cav-mae/src')

with open('/home/ids/hmalard/vggCAVFull.json', 'r') as file:
    data = json.load(file)

print(os.getcwd())
from models import CAVMAE

import dataloader as dataloader


model_path = '/home/ids/hmalard/CAV-MAE-SCALE.pth'
# CAV-MAE model with decoder
audio_model = CAVMAE(audio_length=1024, #\  all models trained with 10s audio
                     modality_specific_depth=11,# \ # all models trained with 11 modality-specific layers and 1 shared layer
                     norm_pix_loss=True, tr_pos=False) # most models are trained with pixel normalization and non-trainabe positional embedding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl_weight = torch.load(model_path, map_location=device)
audio_model = torch.nn.DataParallel(audio_model) # it is important to convert the model to dataparallel object as all weights are saved in dataparallel format (i.e., in module.xxx)
miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=True)
print(miss, unexpected) # check if all weights are correctly loaded


data = '/home/ids/hmalard/vggCAVFull.json'
label_csv = '/home/ids/hmalard/class_labels_indices_vgg.csv'
dataset = 'vggsound'


def get_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return torch.Tensor([cos_sim])

if not isinstance(audio_model, nn.DataParallel):
    audio_model = nn.DataParallel(audio_model)

audio_model = audio_model.to(device)
audio_model.eval()
batch_size = 1
missed = []
simsL =[]
for frame in range(9):
    audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset,
                'mode': 'eval', 'mean': -5.081, 'std': 4.4849, 'noise': False, 'im_res': 224, 'frame_use': frame,'get_id':True}
    val_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(data, label_csv=label_csv, audio_conf=audio_conf), batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)#, collate_fn=lambda x: x)
    
    A_a_feat, A_v_feat = [], []
    sim = []
    idL = []
    with torch.no_grad():
        for i, (a_input, v_input, kj,ids,missing) in enumerate(val_loader):
            audio_input, video_input = a_input.to(device), v_input.to(device)
            #break
            with autocast():
                audio_output, video_output = audio_model.module.forward_feat(audio_input, video_input)
                # mean pool all patches
                audio_output = torch.mean(audio_output, dim=1)
                video_output = torch.mean(video_output, dim=1)
                # normalization
                audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
                video_output = torch.nn.functional.normalize(video_output, dim=-1)
            audio_output = audio_output.to('cpu').detach()
            video_output = video_output.to('cpu').detach()
            missed.append(missing)
            print(i)
            for b in range(audio_output.shape[0]):
                sim.append(get_similarity(audio_output[b],video_output[b]))
                idL.append(ids[b])
        
    simsL.append(torch.cat(sim))
    torch.save(torch.stack(simsL),'/home/ids/hmalard/audible/protected/dev/dcase/studies/010_CAV_MAE_filtering/simMat-frame'+str(frame)+'.pt')
    with open("/home/ids/hmalard/audible/protected/dev/dcase/studies/010_CAV_MAE_filtering/IDs-"+str(frame)+".json", 'w') as f:
        json.dump(idL, f, indent=2) 
    
simL = torch.stack(simsL)
torch.save(torch.stack(simsL),'/home/ids/hmalard/audible/protected/dev/dcase/studies/010_CAV_MAE_filtering/simMat-Full.pt')
with open("/home/ids/hmalard/audible/protected/dev/dcase/studies/010_CAV_MAE_filtering/IDs-Full.json", 'w') as f:
    json.dump(idL, f, indent=2) 