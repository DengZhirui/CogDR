DATASET=$1

python kg/kg_extractor.py --dataset $DATASET_my --mode read_item
python kg/kg_extractor.py --dataset ml-1m_my --name ml-1m- --mode reset_index
python kg/train.py -exp True -mn TransR -ds ml-1m -dsp ../datasets/ml-1m_my/kgdata/ -hpf kg/custom_hp.yaml > transr.out
