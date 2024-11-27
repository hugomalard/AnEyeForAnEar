import argparse
import torch

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

import requests
from PIL import Image
from io import BytesIO
import re

from transformers import ClapModel

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args,n):
    # Model
    disable_torch_init()

    backbone = ClapModel.from_pretrained("laion/clap-htsat-unfused").to('cuda')
    model_name = get_model_name_from_path(args.model_path)
    #model_name='liuhaotian/llava-v1.5-7b-lora'
    args.model_base ='liuhaotian/llava-v1.5-7b'
    #args.model_path = '/home/ids/hmalard/testL/LLaVA/checkpoints/lora_FT_noisy/bad_align_audio_vis_2_llava-lora'
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, None,model_name,audio='laion/clap-htsat-unfused',modality='image',prefix=True
    )
    prefix = torch.load('/home/ids/hmalard/testL/LLaVA/checkpoints/prefixAC/prefix.bin')
    model.load_state_dict(prefix,strict=False)
    model.mask = 0
    model.noise = n
    qs = args.query
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

    image_files = image_parser(args)
    images = load_images(image_files)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    #model.load_state_dict(torch.load('/home/ids/hmalard/testL/LLaVA/checkpoints/audio-vis-soundScene-llava-v1.5-7b/audio_projector_audio_vis.bin'),strict=False)
    #model.load_state_dict(torch.load('/home/ids/hmalard/testL/LLaVA/checkpoints/audio-vis-soundScene-llava-v1.5-7b/audio_projector_audio_vis.bin'),strict=False)

    from transformers import AutoProcessor
    import torchaudio
    import numpy as np

    def wav2fbank(filename, filename2=None, mix_lambda=-1,beats = False):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            #print(waveform.shape)

            '''resampler = torchaudio.transforms.Resample(sr,16000)
            waveform = resampler(waveform)
            if len(waveform.shape)>1:
                waveform = waveform[0,:].unsqueeze(0)'''
            #print(waveform.shape)

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
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
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

    audio_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    audio,sr=torchaudio.load('/home/ids/hmalard/test.wav')
    if sr != 48000:
        resampler = torchaudio.transforms.Resample(sr,48000)
        audio = resampler(audio)
        sr = 48000

    if audio.shape[0]>1:
        audio = audio_processor(audios=np.array(audio[0]), return_tensors="pt", padding=True,sampling_rate=48000)['input_features']
    else:
        audio = audio_processor(audios=np.array(audio), return_tensors="pt", padding=True,sampling_rate=48000)['input_features']

    #audio = audio_processor(audios=np.array(audio[0]), return_tensors="pt", padding=True,sampling_rate=48000)['input_features']
    audios = audio


    #CAV MAE audio loading
    fbank = wav2fbank('/home/ids/hmalard/test.wav')
    audios = (fbank - (-5.081)) / (4.4849)

    #CHANGED JUST FOR CAV MAE NEEDS TO BE UNCOMMENTED
    #model.model.audio_tower.audio_tower = backbone.audio_model.audio_encoder
    model.model.audio_tower.audio_tower.to(dtype=torch.float16)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            #stopping_criteria=[stopping_criteria],
            audios = audios.to(dtype=torch.float16,device='cuda').unsqueeze(0),
            only_im = False
        )
    print('AudioCaps prefix')
    for i in range(len(output_ids)):
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[i].strip()
        print(outputs)


    prefix = torch.load('/home/ids/hmalard/testL/LLaVA/checkpoints/prefixAC2/prefix.bin')
    model.load_state_dict(prefix,strict=False)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            #stopping_criteria=[stopping_criteria],
            audios = audios.to(dtype=torch.float16,device='cuda'),
            only_im = False
        )
    print('300 epochs prefix')
    for i in range(len(output_ids)):
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[i].strip()
        print(outputs)

    prefix = torch.load('/home/ids/hmalard/testL/LLaVA/checkpoints/prefixAC500/prefix.bin')
    model.load_state_dict(prefix,strict=False)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            #stopping_criteria=[stopping_criteria],
            audios = audios.to(dtype=torch.float16,device='cuda'),
            only_im = False
        )
    print('500 epochs prefix')
    for i in range(len(output_ids)):
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[i].strip()
        print(outputs)
    '''input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args,0)
    '''eval_model(args,200)
    eval_model(args,500)
    eval_model(args,600)
    eval_model(args,700)'''