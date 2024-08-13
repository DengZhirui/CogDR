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
python kg/kg_extractor.py --dataset ml-1m_my --mode read_item
python kg/kg_extractor.py --dataset ml-1m_my --name ml-1m- --mode reset_index
python kg/train.py -exp True -mn TransR -ds ml-1m -dsp ../datasets/ml-1m_my/kgdata/ -hpf kg/custom_hp.yaml > transr.out&
python get_u2c.py --dataset ml-1m_my
```

### Recommendation model Training
```
python main.py --model LA_KG_CL_CAT_Adam_0.05 --dataset ml-1m_my --gpu 0 --lr 0.05 --category_balance False --loss_weight 0.05
```
