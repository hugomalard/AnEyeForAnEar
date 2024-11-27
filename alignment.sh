source .bashrc

model=cav-mae 
masking_ratio=0
mask_mode=unstructured
contrast_loss_weight=0.0
mae_loss_weight=0.0
norm_pix_loss=True
tr_pos=False

pretrain_path=./cav-mae-scale++.pth
exp_dir=./expe/ATOM

lr=1e-4
epoch=20
bal=None
lrscheduler_start=10
lrscheduler_decay=0.5
lrscheduler_step=5
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
mixup=0.0
batch_size=256
lr_adapt=False

method=emd
x_attn=True

dataset=audioset
tr_data=./AS500k.json  #PATH_TRAIN_DATA_JSON
te_data=./eval_AS.json #PATH_EVAL_DATA_JSON


CUDA_CACHE_DISABLE=1 python -W ignore ./cav-mae/src/run_distribution_alignment.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./cav-mae/class_labels_indices_AS.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--mixup ${mixup} --bal ${bal} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} --warmup True \
--lr_adapt ${lr_adapt} \
--norm_pix_loss ${norm_pix_loss} \
--pretrain_path ${pretrain_path} \
--mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} \
--tr_pos ${tr_pos} --masking_ratio ${masking_ratio} --mask_mode ${mask_mode} \
--precise_frame False --use_clip True --use_clap False --x_attn ${x_attn} --method ${method} \
