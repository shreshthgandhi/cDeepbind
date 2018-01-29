#!/usr/bin/env bash
nohup python bin/train_model.py --protein RNCMPT00018_full RNCMPT00025_full RNCMPT00032_full --configuration configs/default_config.yml --gpus 0&
nohup python bin/train_model.py --protein RNCMPT00112_full RNCMPT00117_full RNCMPT00126_full --configuration configs/default_config.yml --gpus 1&
nohup python bin/train_model.py --protein RNCMPT00274_full RNCMPT00033_full RNCMPT00172_full --configuration configs/default_config.yml --gpus 2&
nohup python bin/train_model.py --protein RNCMPT00268_full RNCMPT00269_full RNCMPT00046_full --configuration configs/default_config.yml --gpus 3&
nohup python bin/train_model.py --protein RNCMPT00101_full RNCMPT00102_full RNCMPT00103_full --configuration configs/default_config.yml --gpus 4&
nohup python bin/train_model.py --protein RNCMPT00104_full RNCMPT00105_full RNCMPT00047_full --configuration configs/default_config.yml --gpus 5&
nohup python bin/train_model.py --protein RNCMPT00066_full RNCMPT00165_full RNCMPT00121_full --configuration configs/default_config.yml --gpus 6&
nohup python bin/train_model.py --protein RNCMPT00018_full RNCMPT00025_full RNCMPT00032_full --configuration configs/cnn_config.yml --gpus 7&
nohup python bin/train_model.py --protein RNCMPT00112_full RNCMPT00117_full RNCMPT00126_full --configuration configs/cnn_config.yml --gpus 8&
nohup python bin/train_model.py --protein RNCMPT00274_full RNCMPT00033_full RNCMPT00172_full --configuration configs/cnn_config.yml --gpus 9&
nohup python bin/train_model.py --protein RNCMPT00268_full RNCMPT00269_full RNCMPT00046_full --configuration configs/cnn_config.yml --gpus 10&
nohup python bin/train_model.py --protein RNCMPT00101_full RNCMPT00102_full RNCMPT00103_full --configuration configs/cnn_config.yml --gpus 11&
nohup python bin/train_model.py --protein RNCMPT00104_full RNCMPT00105_full RNCMPT00047_full --configuration configs/cnn_config.yml --gpus 12&
nohup python bin/train_model.py --protein RNCMPT00066_full RNCMPT00165_full RNCMPT00121_full --configuration configs/cnn_config.yml --gpus 13&