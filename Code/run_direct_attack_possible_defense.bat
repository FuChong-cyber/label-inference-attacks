python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --multistep_grad True --multistep_grad_bins 6 --use-top-model False
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --multistep_grad True --multistep_grad_bins 12 --use-top-model False
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --multistep_grad True --multistep_grad_bins 18 --use-top-model False
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --multistep_grad True --multistep_grad_bins 24 --use-top-model False
exit

::CIFAR10
::start "gc,0.25"
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --gc True --gc-preserved-percent 0.25 --use-top-model False
::start "gc,0.10"
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --gc True --gc-preserved-percent 0.10 --use-top-model False
::start "gc,0.75"
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --gc True --gc-preserved-percent 0.75 --use-top-model False
::start "gc,0.50"
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --gc True --gc-preserved-percent 0.50 --use-top-model False

::start "ppdl,0.75"
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --ppdl True --ppdl-theta-u 0.75 --use-top-model False
::start "ppdl,0.50"
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --ppdl True --ppdl-theta-u 0.50 --use-top-model False
::start "ppdl,0.25"
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --ppdl True --ppdl-theta-u 0.25 --use-top-model False
::start "ppdl,0.10"
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --ppdl True --ppdl-theta-u 0.10 --use-top-model False

::start "ng,1e-4"
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --lap-noise True --noise-scale 1e-4 --use-top-model False
::start "ng,1e-3"
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --lap-noise True --noise-scale 1e-3 --use-top-model False
::start "ng,1e-2"
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --lap-noise True --noise-scale 1e-2 --use-top-model False
::start "ng,1e-1"
python vfl_framework.py --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --lap-noise True --noise-scale 1e-1 --use-top-model False
