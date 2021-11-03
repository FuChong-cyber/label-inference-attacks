::Yahoo
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d Yahoo --path-dataset D:/Datasets/yahoo_answers_csv/ --k 5 --epochs 25 --lr 1e-3 -b 16 --stone1 15 --stone2 25
python vfl_framework.py --use-mal-optim True --use-mal-optim-all True --use-mal-optim-top False -d Yahoo --path-dataset D:/Datasets/yahoo_answers_csv/ --k 5 --epochs 25 --lr 1e-3 -b 16 --stone1 15 --stone2 25
python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Yahoo --path-dataset D:/Datasets/yahoo_answers_csv/ --k 5 --epochs 25 --lr 1e-3 -b 16 --stone1 15 --stone2 25
::Criteo
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2
python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2
python vfl_framework.py --use-mal-optim True --use-mal-optim-all True --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2
::CINIC10L
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CINIC10L --path-dataset D:/Datasets/CINIC10L --k 4 --epochs 100 --half 16
python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d CINIC10L --path-dataset D:/Datasets/CINIC10L --k 4 --epochs 100 --half 16
python vfl_framework.py --use-mal-optim True --use-mal-optim-all True --use-mal-optim-top False -d CINIC10L --path-dataset D:/Datasets/CINIC10L --k 4 --epochs 100 --half 16
::CIFAR10
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16
python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16
python vfl_framework.py --use-mal-optim True --use-mal-optim-all True --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16
::CIFAR100
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR100 --path-dataset D:/Datasets/CIFAR100  --k 5 --epochs 150 --half 16
python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d CIFAR100 --path-dataset D:/Datasets/CIFAR100  --k 5 --epochs 150 --half 16
python vfl_framework.py --use-mal-optim True --use-mal-optim-all True --use-mal-optim-top False -d CIFAR100 --path-dataset D:/Datasets/CIFAR100  --k 5 --epochs 150 --half 16
::TinyImageNet
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d TinyImageNet --path-dataset D:/Datasets/TinyImageNet   --k 5 --epochs 110 --half 32 -b 128
python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d TinyImageNet --path-dataset D:/Datasets/TinyImageNet   --k 5 --epochs 110 --half 32 -b 128
python vfl_framework.py --use-mal-optim True --use-mal-optim-all True --use-mal-optim-top False -d TinyImageNet --path-dataset D:/Datasets/TinyImageNet   --k 5 --epochs 110 --half 32 -b 128
::IDC(BHI)
python vfl_framework_for_idc.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False --party-num 2
python vfl_framework_for_idc.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False --party-num 2
python vfl_framework_for_idc.py --use-mal-optim True --use-mal-optim-all True --use-mal-optim-top False --party-num 2
::BCW
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d BCW --path-dataset D:/Datasets/BreastCancerWisconsin/wisconsin.csv --k 2 --epochs 30 --lr 1e-2 -b 16 --half 14