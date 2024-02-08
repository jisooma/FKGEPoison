# 三、WNRR
# 1、服务器执行恶意攻击
# 'TransE', 'RotatE', 'DistMult', 'ComplEx'
# （1）训练
c='cuda:0'

python ../main.py --state_dir ./state_attack_dpas --log_dir ./log_attack_dpas --tb_log_dir ./tb_log_attack_dpas --setting DPA_S --mode train \
    --num_client 3 --dataset_name WNRR  --client_model DistMult  --server_model DistMult --gpu $c \
    --attack_entity_ratio 10
python ../main.py --state_dir ./state_attack_dpas --log_dir ./log_attack_dpas --tb_log_dir ./tb_log_attack_dpas --setting DPA_S --mode test \
   --num_client 3   --dataset_name WNRR  --client_model DistMult  --server_model DistMult --gpu $c \
   --attack_entity_ratio 10


#python ../main.py --state_dir ./state_attack_fmpas --log_dir ./log_attack_fmpas --tb_log_dir ./tb_log_attack_fmpas --setting FMPA_S --mode train \
#    --num_client 3 --dataset_name WNRR  --client_model DistMult  --server_model DistMult --gpu $c \
#    --attack_entity_ratio 10
#python ../main.py --state_dir ./state_attack_fmpas --log_dir ./log_attack_fmpas --tb_log_dir ./tb_log_attack_fmpas --setting FMPA_S --mode test \
#   --num_client 3   --dataset_name WNRR  --client_model DistMult  --server_model DistMult --gpu $c \
#   --attack_entity_ratio 10
#
#
#python ../main.py --state_dir ./state_attack_cpa --log_dir ./log_attack_cpa --tb_log_dir ./tb_log_attack_cpa --setting CPA --mode train \
#    --num_client 3 --dataset_name WNRR  --client_model DistMult  --server_model DistMult --gpu $c \
#    --attack_entity_ratio 10
#python ../main.py --state_dir ./state_attack_cpa --log_dir ./log_attack_cpa --tb_log_dir ./tb_log_attack_cpa --setting CPA --mode test \
#   --num_client 3   --dataset_name WNRR  --client_model DistMult  --server_model DistMult --gpu $c \
#   --attack_entity_ratio 10

#python ../main.py --state_dir ./state_clean --log_dir ./log_clean --tb_log_dir ./tb_log_clean --setting FedE --mode train  \
#      --num_client 3 --dataset_name WNRR    --client_model DistMult  --server_model DistMult --gpu $c \
#      --attack_entity_ratio 10
#python ../main.py --state_dir ./state_clean --log_dir ./log_clean --tb_log_dir ./tb_log_clean --setting FedE --mode test \
#     --num_client 3 --dataset_name WNRR     --client_model DistMult  --server_model DistMult --gpu $c \
#     --attack_entity_ratio 10
   
