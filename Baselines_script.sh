# SLIME4Rec
python run_seq.py --dataset='beauty' --train_batch_size=256  --model='SLIME4Rec' --eval_epoch=-1 --shuffle=True  --gpu_id=0 --contrast='us_x' --hidden_dropout_prob=0.5 --freq_dropout_prob=0.5 --filter_mixer='M' --dynamic_ratio=0.4  --SSL_AUG=DuoRec

# BSARec
python run_seq.py --dataset='beauty' --train_batch_size=256  --model='BSARec' --eval_epoch=-1 --shuffle=True  --gpu_id=0 --contrast=None --hidden_dropout_prob=0.5 --freq_dropout_prob=0.5 --alpha=0.7

# FMLPRec
python run_seq.py --dataset='beauty' --train_batch_size=256  --model='FMLPRec' --eval_epoch=-1 --shuffle=True  --gpu_id=0 --contrast=None --hidden_dropout_prob=0.5 --freq_dropout_prob=0.6

# DuoRec
python run_seq.py --dataset='beauty' --train_batch_size=256  --model='DuoRec' --eval_epoch=-1 --shuffle=True  --gpu_id=0 --contrast='us_x' --hidden_dropout_prob=0.5 --attn_dropout_prob=0.5 --SSL_AUG='DuoRec'

# SASRec
python run_seq.py --dataset='beauty' --train_batch_size=256  --model='SASRec' --eval_epoch=-1 --shuffle=True  --gpu_id=0 --contrast=None --hidden_dropout_prob=0.5 --attn_dropout_prob=0.5