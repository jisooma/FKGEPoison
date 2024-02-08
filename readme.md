# Poisoning Attack on Federated Knowledge Graph Embedding  (WWW2024)

- This repository contains the simplified code for the paper:  [Poisoning Attack on Federated Knowledge Graph Embedding](https://openreview.net/forum?id=6qncjuadJW).

- This paper is the first work to systematise the risks of FKGE poisoning attacks, from which we develop a novel framework for poisoning attacks that force the victim to predict specific false facts.  

  ![](https://raw.githubusercontent.com/mazhixiu09/pictures/master/blogimg/202402071940087.png)
  
  
  
  
  
  

## Requirements

```bash
python=3.8.15
pytorch=1.12.0
numpy=1.23.5
tqdm=4.65.0
```



## Attacks

### 0、Generate client dataset

```bash
bash ./process_data/generate_client_data.sh
```



### 1、FMPA-S

- The Fixed Model Poisoning  Attack

```bash
python ./main.py --state_dir ./state_attack_fmpas --log_dir ./log_attack_fmpas --tb_log_dir ./tb_log_attack_fmpas --setting FMPA_S --mode train \
    --num_client 3 --dataset_name WNRR  --client_model TransE  --server_model TransE --gpu 'cuda:0' \
    --attack_entity_ratio 10
python ./main.py --state_dir ./state_attack_fmpas --log_dir ./log_attack_fmpas --tb_log_dir ./tb_log_attack_fmpas --setting FMPA_S --mode test \
   --num_client 3   --dataset_name WNRR  --client_model TransE  --server_model TransE --gpu 'cuda:0' \
   --attack_entity_ratio 10
```



### 2、DPA-S

- The Dynamic Poisoning  Attack

```bash
python ./main.py --state_dir ./state_attack_dpas --log_dir ./log_attack_dpas --tb_log_dir ./tb_log_attack_dpas --setting DPA_S --mode train \
    --num_client 3 --dataset_name WNRR  --client_model TransE  --server_model TransE --gpu 'cuda:0' \
    --attack_entity_ratio 10
    
python ./main.py --state_dir ./state_attack_dpas --log_dir ./log_attack_dpas --tb_log_dir ./tb_log_attack_dpas --setting DPA_S --mode test \
   --num_client 3   --dataset_name WNRR  --client_model TransE  --server_model TransE --gpu 'cuda:0' \
   --attack_entity_ratio 10
```



### 3、CPA

- The Client Poisoning Attack

```bash
python ./main.py --state_dir ./state_attack_cpa --log_dir ./log_attack_cpa --tb_log_dir ./tb_log_attack_cpa --setting CPA --mode train \
    --num_client 3 --dataset_name WNRR  --client_model TransE  --server_model TransE --gpu 'cuda:0' \
    --attack_entity_ratio 10
    
python ./main.py --state_dir ./state_attack_cpa --log_dir ./log_attack_cpa --tb_log_dir ./tb_log_attack_cpa --setting CPA --mode test \
   --num_client 3   --dataset_name WNRR  --client_model TransE  --server_model TransE --gpu 'cuda:0' \
   --attack_entity_ratio 10
```



### 4、FedE

- [FedE: Embedding Knowledge Graphs in Federated Setting](https://dl.acm.org/doi/fullHtml/10.1145/3502223.3502233).

```bash
python ./main.py --state_dir ./state_clean --log_dir ./log_clean --tb_log_dir ./tb_log_clean --setting FedE --mode train  \
      --num_client 3 --dataset_name WNRR --client_model TransE  --server_model TransE --gpu 'cuda:0' \
      --attack_entity_ratio 10

python ./main.py --state_dir ./state_clean --log_dir ./log_clean --tb_log_dir ./tb_log_clean --setting FedE --mode test \
     --num_client 3 --dataset_name WNRR    --client_model TransE  --server_model TransE --gpu 'cuda:0' \
     --attack_entity_ratio 10
```



## Citation

- If you find this code useful in your research, please cite:

```bash
@inproceedings{
  title={Poisoning Attack on Federated Knowledge Graph Embedding},
  author={Enyuan Zhou, Song Guo, Zhixiu Ma, Zicong Hong, Tao GUO, Peiran Dong},
  year={2024}
}
```

#  

























