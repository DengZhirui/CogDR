
DATASET=`ml-10M`
cd data_process
python data_process.py --dataset $DATASET
cd ../KG_LGCN_COMBINE
python kg/kg_extractor.py --dataset ml-10M_my --mode read_item
python kg/kg_extractor.py --dataset ml-10M_my --name ml-10M- --mode reset_index
conda activate pykg2vec
nohup python kg/train.py -exp True -mn TransR -ds ml-10M -dsp ../datasets/ml-10M_my/kgdata/ -hpf kg/custom_hp.yaml > transr.out&
python get_u2c.py --dataset ml-10M_my
nohup python main.py --dataset ml-10M_my --model KG_LGCN_COMBINE --lr 0.05 --step 1 --gpu 2 --cl_bs 1024 --batch_size 2048 &
nohup python main.py --dataset ml-10M_my --model KG_LGCN_CL --lr 0.05 --gpu 0 --cl_bs 512 --batch_size 2048 --loss_weight 0 --temperature 0.1 --epoch 500 --topk 0 &
nohup python main.py --dataset Video_Games_my --model KG_CAT_FFT2 --lr 0.01 --gpu 1 --cl_bs 1024 --batch_size 2048 --loss_weight 0.02 --topk 5 > videogames.out&
nohup python main.py --model dgrec_KG_Adam_0.005 --dataset Video_Games_my --gpu 1 --lr 0.05 --category_balance False --loss_weight 0.005&
nohup python main.py --model LA_KG_CL_CAT_Adam_0.05 --dataset Musical_Instruments_my --gpu 0 --lr 0.05 --category_balance False --loss_weight 0.05&


###LLM
python kgemb_topk.py --dataset ml-1m_my --k 100
python ll_data_process.py --dataset ml-1m_my


###DGCN
cd DGCN
conda activate dgcn
python dgcn_data_process.py --dataset ml-10M_my
python app.py --flagfile ./config/xxx.cfg

###Popularity
cd Popularity
python main.py --dataset ml-10M_my


###DGRec
cd DGRec
nohup python main.py --dataset ml-10M_my --gpu 0 --lr 0.05&

###LightGCN
cd LightGCN_our
nohup python main.py --dataset ml-10M_my --gpu 0 --lr 0.05&

