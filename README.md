A Pretraining Numerical Reasoning Model for Ordinal Constrained KBQA 
====
This is the code for the paper:
    [A Pretraining Numerical Reasoning Model for Ordinal Constrained
Question Answering on Knowledge Base](https://aclanthology.org/2021.findings-emnlp.159/) in EMNLP 2021 Findings.

An advanced model to deal with the same problem can be found in the following paper: 
    [Injecting Numerical Reasoning Skills into Knowledge Base Question Answering Models](https://arxiv.org/abs/2112.06109) in arxiv preprint.

Requirements
===
- Python 3.8
- Pytorch >= 1.6

Dataset
===
Download preprocessed datasets from [google drive], and unzip it into dataset folder.

Reasoning
===
To train and evaluate the reasoning model, change to directory ./NumReasoning. You can also download checkpoints from [here](https://drive.google.com/drive/folders/1jFcV1vj6wbAVuIOMyjgULZNkLBXjmDPa?usp=sharing), and unzip it into checkpoint folder.


Basic Reasoning
---
To test the original reasoning model.

    python main_nsm.py --model_name gnn --data_folder ../CWQ/ --checkpoint_dir ../checkpoint/CWQ_num/ --experiment_name eval_CWQ_gnn_num_50epoch
    --entity2id entities_expanded.txt --batch_size 40 --test_batch_size 40 --num_step 4 --entity_dim 50 --word_dim 300 --node_dim 50 --eval_every 1 
    --eps 0.95 --num_epoch 50 --use_self_loop --lr 1e-4 --q_type seq --word_emb_file word_emb_300d.npy --reason_kb --encode_type --loss_type kl 
    --load_experiment CWQ_nsm-h1.ckpt --is_eval

Num Reasoning
---
To train the Num reasoning model.

    python main_nsm.py --model_name gnn --data_folder ../CWQ/ --checkpoint_dir ../checkpoint/CWQ_num/ --experiment_name CWQ_gnn_num_50epoch
    --entity2id entities_expanded.txt --batch_size 40 --test_batch_size 40 --num_step 4 --entity_dim 50 --word_dim 300 --node_dim 50 --eval_every 1  
    --eps 0.95 --num_epoch 50 --use_self_loop --lr 1e-4 --q_type seq --word_emb_file word_emb_300d.npy --reason_kb --encode_type --loss_type kl 
    --use_num --use_nsm_num --relation_embedding_file ../CWQ/cwq_rel_embedding.npy -
    -load_num CWQ_num_model.pth --load_experiment CWQ_nsm-h1.ckpt

To test the Num reasoning model.

    python main_nsm.py --model_name gnn --data_folder ../CWQ/ --checkpoint_dir ../checkpoint/CWQ_num/ --experiment_name eval_CWQ_gnn_num_50epoch
    --entity2id entities_expanded.txt --batch_size 40 --test_batch_size 40 --num_step 4 --entity_dim 50 --word_dim 300 --node_dim 50 --eval_every 1  
    --eps 0.95 --num_epoch 50 --use_self_loop --lr 1e-4 --q_type seq --word_emb_file word_emb_300d.npy --reason_kb --encode_type --loss_type kl 
    --use_num --use_nsm_num --relation_embedding_file ../CWQ/cwq_rel_embedding.npy 
    --load_num CWQ_num_model.pth --load_experiment CWQ_gnn_num_50epoch-h1.ckpt --is_eval 
