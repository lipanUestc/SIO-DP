## 1、Train Reference Models

```bash
python cifar10.py --disable-dp --checkpoint-file cifar10_reference_model_1 --seed 1
```

## 2、Estimate Complexity Matrix

```bash
python estimate_complexity.py --dataset cifar10 --reference_model_num 5 
```

## 3、Bootstrap Resampling

```bash
python bootstrap_resampling.py
```

## 4、Train DP Models

```bash
python cifar10.py --sigma 1.0 --checkpoint-file cifar10_eps_1.0_model --batch-size 256 --epochs 100 --seed 0
```
 
