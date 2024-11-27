python ./LLaVA/eval_captions_AC.py --model-path ../expe/DALI_OT_ATT/models \
    --modality audio \
    --use_prefix false --use_mlp true \
    --audio_path /tsi/audiosig/audible/dcase/data/AudioSet_Resampled/ \
    --frame_path /tsi/audiosig/audible/dcase/data/AudioSet_Frames_CAV/