# An Eye for an Ear: Zero-shot Audio Description Leveraging an Image Captioner using Audiovisual Distribution Alignment

Here is the repository of [*An eye for an Ear*].
It contains all the files to reproduce our decider training and evaluate it, as well as our best model. 

<p align="center">
	<img src="scheme.png" width="500" height="400">
</p>

### Preprocessing
Audio and images should be preprocessed following the CAV-MAE repository.
A json file containing the AudioSet file paths should be formated as CAV-MAE repository too.
Download the model weights (backbone and MLP) for inference and put them in ./expe/DALI_OT_ATT, or the csv file containing the subset of AudioSet that we used for the ditribution alignment training [there](https://zenodo.org/records/13941386?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVjZTI2ZTkwLWNkOTEtNDJjNS1iNjUyLTE2YTEzYjFlMzBmOCIsImRhdGEiOnt9LCJyYW5kb20iOiJiMTgzZmRhOWZlNzYxMTJlNzI3YmFjMWFhZDYxOWExNyJ9.DDHjUSCNdZYpZWrEnVTGKEvX0w2yfJfYhnYmacZj_96IlCMtHPDbamBkAnRXbaNFLCpAPVIsfUHS__kftsidYQ)

## Environment creation
'''
conda create -n dali python=3.10 -y
conda activate dali
pip install --upgrade pip 
pip install -e ./LLaVA/.
pip install -e ./LLaVA/.[train]
pip install flash-attn==2.6.3 --no-build-isolation
pip install -r requirements.txt
'''

## Training
### Stage 1a: prefix tuning
The first training step consists in training additional prompt tokens to specify the task to the model: audio captioning.
They are trained in a zero shot manner, without relying on audios. They instead used images from AudioSet and their associated audio caption from AudioCaps.

'''
./prefix_tuning.sh
'''

### Stage 1b: Distribution alignment
In parallel to the stage 1a, an audio backbone is trained to align the audio *token distribution* with the image encoder of Llava. This can be done using Maximum Mean Discreptancy of Attentive Optimal Transport.
'''
./alignment.sh 
'''

### Stage 2: Audio-Visual Distillation
Once the prefix are tuned and the backbones aligned, we feed all the tokens (prefix,image and audio tokens) of given image-audio pair to the LLM and generate pseudo-caption. They are then used to fine-tune (in audio-only) the MLP that maps the audio tokens to the LLM embedding space.
'''
./infere_pseudo_labels.sh
./audio_vis_distil.sh
'''

## Inference
Inference on AudioCaps test set can be performed using the following command:
'''
./inference.sh
'''
Note that you can change the input modality by changing the --modality (audio,audio_vis or image).