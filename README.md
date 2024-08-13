# CogDR

Utilizing dataset **ml-1m** as example.

### Data Pre-Process
```
cd data_process
python data_process.py --dataset ml-1m
```

### Embedding Initializing
```
cd code
sh emb_initializing.sh ml-1m
python get_u2c.py --dataset ml-1m_my
```

### Recommendation model Training
```
python main.py --model LA_KG_CL_CAT_Adam_0.05 --dataset ml-1m_my --gpu 0 --lr 0.05 --category_balance False --loss_weight 0.05
```
