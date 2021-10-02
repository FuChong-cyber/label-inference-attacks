python model_completion.py --dataset-name CIFAR10 --dataset-path D:/Datasets/CIFAR10  --n-labeled 40 --party-num 2 --half 16 --k 4 --resume-name CIFAR10_saved_framework_lr=0.1_mal_multistep_grad_bins=6_half=16.pth --print-to-txt 1 --epochs 25
exit
python model_completion.py --dataset-name CIFAR10 --dataset-path D:/Datasets/CIFAR10  --n-labeled 40 --party-num 2 --half 16 --k 4 --resume-name CIFAR10_saved_framework_lr=0.1_mal_multistep_grad_bins=12_half=16.pth --print-to-txt 1 --epochs 25
python model_completion.py --dataset-name CIFAR10 --dataset-path D:/Datasets/CIFAR10  --n-labeled 40 --party-num 2 --half 16 --k 4 --resume-name CIFAR10_saved_framework_lr=0.1_mal_multistep_grad_bins=18_half=16.pth --print-to-txt 1 --epochs 25
python model_completion.py --dataset-name CIFAR10 --dataset-path D:/Datasets/CIFAR10  --n-labeled 40 --party-num 2 --half 16 --k 4 --resume-name CIFAR10_saved_framework_lr=0.1_mal_multistep_grad_bins=24_half=16.pth --print-to-txt 1 --epochs 25


exit
::defense - multi-step grad
start "multistep,bins=6,mc,criteo" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_multistep_grad_bins=6_half=4096.pth --print-to-txt 1 --epochs 5
start "multistep,bins=12,mc,criteo" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_multistep_grad_bins=12_half=4096.pth --print-to-txt 1 --epochs 5
start "multistep,bins=18,mc,criteo" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_multistep_grad_bins=18_half=4096.pth --print-to-txt 1 --epochs 5
start "multistep,bins=24,mc,criteo" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_multistep_grad_bins=24_half=4096.pth --print-to-txt 1 --epochs 5
pause

python model_completion.py --dataset-name BC_IDC --dataset-path D:/Datasets/BC_IDC --n-labeled 70 --party-num 2 --half 1 --k 2 --resume-name BC_IDC_saved_framework_lr=0.05_mal_multistep_grad_bins=6_party-num=2.pth --print-to-txt 1 --epochs 10 --batch-size 32
python model_completion.py --dataset-name BC_IDC --dataset-path D:/Datasets/BC_IDC --n-labeled 70 --party-num 2 --half 1 --k 2 --resume-name BC_IDC_saved_framework_lr=0.05_mal_multistep_grad_bins=12_party-num=2.pth --print-to-txt 1 --epochs 10 --batch-size 32
python model_completion.py --dataset-name BC_IDC --dataset-path D:/Datasets/BC_IDC --n-labeled 70 --party-num 2 --half 1 --k 2 --resume-name BC_IDC_saved_framework_lr=0.05_mal_multistep_grad_bins=18_party-num=2.pth --print-to-txt 1 --epochs 10 --batch-size 32
python model_completion.py --dataset-name BC_IDC --dataset-path D:/Datasets/BC_IDC --n-labeled 70 --party-num 2 --half 1 --k 2 --resume-name BC_IDC_saved_framework_lr=0.05_mal_multistep_grad_bins=24_party-num=2.pth --print-to-txt 1 --epochs 10 --batch-size 32
exit

::Criteo
start "gc,0.25,model completion" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_gc-preserved_percent=0.25_half=4096.pth --print-to-txt 1 --epochs 5
start "gc,0.10,model completion" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_gc-preserved_percent=0.1_half=4096.pth --print-to-txt 1 --epochs 5

start "ppdl,0.75,model completion" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_ppdl-theta_u=0.75_half=4096.pth --print-to-txt 1 --epochs 5
start "ppdl,0.50,model completion" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_ppdl-theta_u=0.5_half=4096.pth --print-to-txt 1 --epochs 5
start "ppdl,0.25,model completion" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_ppdl-theta_u=0.25_half=4096.pth --print-to-txt 1 --epochs 5
start "ppdl,0.10,model completion" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_ppdl-theta_u=0.1_half=4096.pth --print-to-txt 1 --epochs 5

exit

start "ng,1e-4,model completion" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_lap_noise-scale=0.0001_half=4096.pth --print-to-txt 1 --epochs 5
start "ng,1e-3,model completion" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_lap_noise-scale=0.001_half=4096.pth --print-to-txt 1 --epochs 5
start "ng,1e-2,model completion" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_lap_noise-scale=0.01_half=4096.pth --print-to-txt 1 --epochs 5
start "ng,1e-1,model completion" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_lap_noise-scale=0.1_half=4096.pth --print-to-txt 1 --epochs 5

start "gc,0.75,model completion" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_gc-preserved_percent=0.75_half=4096.pth --print-to-txt 1 --epochs 5
start "gc,0.50,model completion" python model_completion.py --dataset-name Criteo --dataset-path D:/Datasets/Criteo/criteo.csv --n-labeled 100 --party-num 2 --half 4096 --k 2 --resume-name Criteo_saved_framework_lr=0.05_mal_gc-preserved_percent=0.5_half=4096.pth --print-to-txt 1 --epochs 5
