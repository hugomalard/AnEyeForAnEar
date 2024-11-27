import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoProcessor, ClapModel

from transformers import AutoFeatureExtractor, ClapModel
import sys
sys.path.append('../cav-mae/src')
import models
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

class CLAPAudioTower(nn.Module):
    def __init__(self, audio_tower,args, delay_load=False,cav=False):
        super().__init__()

        self.is_loaded = False
        self.prefix = torch.nn.Parameter(data=torch.normal(0,0.002,([16,4096])), requires_grad=True)
        self.audio_tower_name = audio_tower
        #self.select_layer = args.mm_vision_select_layer
        #self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.cav = cav
        self.load_model()
       

    def load_model(self):
        if not self.cav:
            print('NOT USING CAV')
            print('PROOF',self.cav)
            self.audio_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
            self.audio_tower = ClapModel.from_pretrained("laion/clap-htsat-unfused").audio_model.audio_encoder #CLIPVisionModel.from_pretrained(self.vision_tower_name)
            self.audio_tower.batch_norm = torch.nn.Identity()
            self.audio_tower.requires_grad_(False)
        else:
            self.audio_tower = models.CAVMAE(audio_length=1024, norm_pix_loss=True, modality_specific_depth=11, tr_pos=False,use_clip=True,use_clap=False,match_hist=True)
            self.audio_tower.blocks_v = None
            self.audio_tower.vision_tower = None
            self.audio_tower.decoder_blocks = None
            print('using CAV audio tower')
        self.is_loaded = True
        self.audio_tower.cav = self.cav
  
        '''random_audio = torch.rand((1,48000))
        import numpy as np
        random_audio=random_audio.to(dtype=torch.float32)
        print('type',random_audio.type())
        audio = self.audio_processor(audios=np.array(random_audio[0]), return_tensors="pt", padding=True,sampling_rate=48000)['input_features']
        print('audio shape ',audio.shape)
        print('audio type',audio.type())
        audio_features = self.audio_tower.get_audio_features(audio.to('cuda'))

        print('worked \n')'''
        '''self.audio_tower.to(dtype=torch.float16)
        inputs = self.audio_processor(random_audio, return_tensors="pt").to(dtype=torch.float16)
        feats = torch.flatten(self.audio_tower.audio_model.audio_encoder(**inputs).last_hidden_state,2)
        print('worked in f16')'''


    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, audios):
        if type(audios) is list:
            audio_features = []
            for audio in audios:
                audio_feature = self.audio_tower.audio_encoder(audios.to(device=self.device))
                audio_feature = torch.flatten(audio_features, 2)
                audio_features.append(audio_feature)
        else:
            #audios = audios.to(dtype=torch.float32,device='cuda') #to(device='cuda').half()#self.device)
            #feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
            #random_audio = torch.rand((100000))
            #print('type \n \n',random_audio.type())
            #self.audio_tower = self.audio_tower.to(dtype=torch.float32)
            '''inputs = feature_extractor(random_audio, return_tensors="pt")
            audio_features = self.audio_tower(audios).last_hidden_state #model.get_audio_features(**inputs)
            print('WORKED FEATS \n \n \n',audio_features.shape)'''

            #self.audio_tower = self.audio_tower.to(dtype=torch.float32)
            #print('Audio type',audios.type())
            #self.audio_tower =  self.audio_tower.half()
            #print('model type ',self.audio_tower.audio_model.audio_encoder.parameters().type())
            #print('class type ',self.audio_tower.audio_model.type())
            if not self.cav:
                audio_features = self.audio_tower(audios).last_hidden_state #audio_model.audio_encoder(audios,return_dict=False)
                #image_forward_outs = self.audio_tower(audios.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                audio_features = torch.flatten(audio_features, 2)
                #image_features = self.feature_select(image_forward_outs).to(audios.dtype)
            else:
                audio_features = self.audio_tower.forward_greedy(audios)
                #audio_features = self.audio_tower.forward_audio_feat(audios)
                #audio_features = self.audio_tower.proj(audio_features)
                #audio_features = self.audio_tower.audio2clip(audio_features)
      
        return audio_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
