# Beauty
python run_seq.py --dataset='beauty' --train_batch_size=256  --model='Muffin' --eval_epoch=-1 --shuffle=True --alpha=0.1 --beta=0.2 --kernel_size=7 --hidden_dropout_prob=0.4 --freq_dropout_prob=0.4 --num_experts=6

# # ML-1M
# python run_seq.py --dataset='ml-1m' --train_batch_size=256  --model='Muffin' --eval_epoch=-1 --shuffle=True --alpha=0.1 --beta=0.2 --kernel_size=3 --hidden_dropout_prob=0.1  --freq_dropout_prob=0.1 --num_experts=4

# # Toys
# python run_seq.py --dataset='toys' --train_batch_size=256  --model='Muffin' --eval_epoch=-1 --shuffle=True  --alpha=0.05 --beta=1 --kernel_size=7 --hidden_dropout_prob=0.4 --freq_dropout_prob=0.4 --num_experts=6 

# # Sports
# python run_seq.py --dataset='sports' --train_batch_size=256  --model='Muffin' --eval_epoch=-1  --shuffle=True --alpha=0.5 --beta=0.05  --kernel_size=7 --hidden_dropout_prob=0.5 --freq_dropout_prob=0.5 --num_experts=6 

# # Yelp
# python run_seq.py --dataset='yelp' --train_batch_size=256 --model='Muffin' --eval_epoch=-1 --shuffle=True --alpha=0.5 --beta=1 --kernel_size=5 --hidden_dropout_prob=0.4  --freq_dropout_prob=0.4 --num_experts=6 

