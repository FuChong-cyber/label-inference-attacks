::Criteo
start "criteo,bins=4" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --multistep_grad True --multistep_grad_bins 4 --multistep_grad_bound_abs 1e-3
start "criteo,bins=2" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --multistep_grad True --multistep_grad_bins 2 --multistep_grad_bound_abs 1e-3
start "criteo,bins=6" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --multistep_grad True --multistep_grad_bins 6 --multistep_grad_bound_abs 1e-3
start "criteo,bins=12" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --multistep_grad True --multistep_grad_bins 12 --multistep_grad_bound_abs 1e-3
start "criteo,bins=18" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --multistep_grad True --multistep_grad_bins 18 --multistep_grad_bound_abs 1e-3
start "criteo,bins=24" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --multistep_grad True --multistep_grad_bins 24 --multistep_grad_bound_abs 1e-3
pause
::BHI
python vfl_framework_for_idc.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False --party-num 2  --multistep_grad True --multistep_grad_bins 6
python vfl_framework_for_idc.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False --party-num 2  --multistep_grad True --multistep_grad_bins 12
python vfl_framework_for_idc.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False --party-num 2  --multistep_grad True --multistep_grad_bins 18
python vfl_framework_for_idc.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False --party-num 2  --multistep_grad True --multistep_grad_bins 24
exit

::CIFAR10
::python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --multistep_grad True --multistep_grad_bins 4
python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --multistep_grad True --multistep_grad_bins 6
python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --multistep_grad True --multistep_grad_bins 12
python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --multistep_grad True --multistep_grad_bins 18
python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d CIFAR10 --path-dataset D:/Datasets/CIFAR10  --k 4 --epochs 100 --half 16 --multistep_grad True --multistep_grad_bins 24

exit

::Criteo
start "gc,0.25" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --gc True --gc-preserved-percent 0.25
start "gc,0.10" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --gc True --gc-preserved-percent 0.10

start "ppdl,0.75" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --ppdl True --ppdl-theta-u 0.75
start "ppdl,0.50" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --ppdl True --ppdl-theta-u 0.50
start "ppdl,0.25" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --ppdl True --ppdl-theta-u 0.25
start "ppdl,0.10" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --ppdl True --ppdl-theta-u 0.10

exit

start "ng,1e-4" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --lap-noise True --noise-scale 1e-4
start "ng,1e-3" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --lap-noise True --noise-scale 1e-3
start "ng,1e-2" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --lap-noise True --noise-scale 1e-2
start "ng,1e-1" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --lap-noise True --noise-scale 1e-1

start "gc,0.75" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --gc True --gc-preserved-percent 0.75
start "gc,0.50" python vfl_framework.py --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2 --gc True --gc-preserved-percent 0.50
