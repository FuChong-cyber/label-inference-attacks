python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 1 --half 4096 --momentum 0.5 --lr 5e-2 --use-top-model False
exit
python vfl_framework_for_idc.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False --party-num 2 --use-top-model False
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d TinyImageNet --path-dataset D:/Datasets/TinyImageNet   --k 5 --epochs 1 --half 32 --use-top-model False
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10 --k 4 --epochs 1 --half 16 --use-top-model False
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CINIC10L --path-dataset D:/Datasets/CINIC10L --k 4 --epochs 1 --half 16 --use-top-model False
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR100 --path-dataset D:/Datasets/CIFAR100  --k 5 --epochs 1 --half 16 --use-top-model False
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 1 --half 4096 --momentum 0.5 --lr 5e-2 --use-top-model False
pause