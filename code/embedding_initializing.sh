DATASET=$1

python emb_init/extractor.py --dataset ${DATASET}_my --mode read_item
python emb_init/extractor.py --dataset ${DATASET}_my --name ${DATASET}- --mode reset_index
python emb_init/train.py -exp True -mn TransR -ds ${DATASET} -dsp ../datasets/${DATASET}_my/kgdata/ -hpf kg/custom_hp.yaml
