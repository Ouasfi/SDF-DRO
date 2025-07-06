head -20 /home/amine/data/ShapeNet/04379243/test_optim.lst | xargs -I {} -P 2 bash -c 'CUDA_VISIBLE_DEVICES=1 python train.py configs/sn_config.json --shapename "04379243/{}/"'
