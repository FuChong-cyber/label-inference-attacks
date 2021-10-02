::Criteo
start "1024-upperbound" python upper_bound_testing.py -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 1024 --momentum 0.5 --lr 5e-2
start "2048-upperbound" python upper_bound_testing.py -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 2048 --momentum 0.5 --lr 5e-2
start "3072-upperbound" python upper_bound_testing.py -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 3072 --momentum 0.5 --lr 5e-2
start "4096-upperbound" python upper_bound_testing.py -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 4096 --momentum 0.5 --lr 5e-2
start "5120-upperbound" python upper_bound_testing.py -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 5120 --momentum 0.5 --lr 5e-2
start "6144-upperbound" python upper_bound_testing.py -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 6144 --momentum 0.5 --lr 5e-2
start "7168-upperbound" python upper_bound_testing.py -d Criteo --path-dataset D:/Datasets/Criteo/criteo.csv --k 2 --epochs 7 --half 7168 --momentum 0.5 --lr 5e-2
exit
::TinyImageNet
python upper_bound_testing.py -d TinyImageNet --path-dataset D:/Datasets/TinyImageNet   --k 5 --epochs 110 --half 8 -b 128
python upper_bound_testing.py -d TinyImageNet --path-dataset D:/Datasets/TinyImageNet   --k 5 --epochs 110 --half 16 -b 128
python upper_bound_testing.py -d TinyImageNet --path-dataset D:/Datasets/TinyImageNet   --k 5 --epochs 110 --half 24 -b 128
python upper_bound_testing.py -d TinyImageNet --path-dataset D:/Datasets/TinyImageNet   --k 5 --epochs 110 --half 32 -b 128
python upper_bound_testing.py -d TinyImageNet --path-dataset D:/Datasets/TinyImageNet   --k 5 --epochs 110 --half 40 -b 128
python upper_bound_testing.py -d TinyImageNet --path-dataset D:/Datasets/TinyImageNet   --k 5 --epochs 110 --half 48 -b 128
python upper_bound_testing.py -d TinyImageNet --path-dataset D:/Datasets/TinyImageNet   --k 5 --epochs 110 --half 56 -b 128
