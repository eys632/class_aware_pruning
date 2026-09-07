# MNIST Structured Neuron Pruning — v2

이 프로젝트는 첫 논문 실험을 위한 **작고 해석 가능한 검증 파이프라인**입니다.

## 서버 권장 환경

현재 확인된 서버:

- Ubuntu 22.04
- 2 × NVIDIA GeForce RTX 5090 (약 32 GB each)
- GPU 1 전용 사용 가능
- NVIDIA driver 580.65.06
- `nvidia-smi`: CUDA 13.0 driver support
- local `nvcc`: CUDA 12.8
- 2 × Xeon Silver 4510, 48 logical CPUs
- RAM 125 GiB

`base` conda 환경은 건드리지 않고 별도 환경을 만드는 것을 권장합니다.

## 1. 환경 생성

```bash
conda create -n mnist-prune python=3.12 -y
conda activate mnist-prune

python -m pip install --upgrade pip

python -m pip install \
  torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130

python -m pip install numpy pandas matplotlib
```

## 2. GPU 1 검증

```bash
CUDA_VISIBLE_DEVICES=1 python verify_gpu.py
```

정상이면 출력에 다음이 나와야 합니다.

- CUDA available: True
- Visible GPUs: 1
- RTX 5090
- matrix multiplication: OK

`CUDA_VISIBLE_DEVICES=1`을 사용했기 때문에 물리 GPU 1은 Python 안에서 `cuda:0`으로 보입니다.

## 3. Smoke test

```bash
CUDA_VISIBLE_DEVICES=1 python experiment.py \
  --seeds 0 \
  --epochs 3 \
  --prune-ratios 0.2 0.5 \
  --out-dir runs/smoke
```

## 4. 첫 본 실험

```bash
CUDA_VISIBLE_DEVICES=1 python experiment.py \
  --seeds 0 1 2 3 4 \
  --epochs 25 \
  --batch-size 2048 \
  --prune-ratios 0.1 0.2 0.3 0.4 0.5 \
  --out-dir runs/global_5seeds
```

### 왜 batch 2048인가?

MNIST MLP는 RTX 5090에 비해 매우 작습니다. 더 큰 batch로 GPU utilization을 높이는 것이 연구 목표가 아니며,
너무 큰 batch는 한 epoch의 optimizer update 수를 과도하게 줄입니다.
2048은 GPU overhead를 낮추면서도 충분한 update 횟수를 확보하기 위한 시작값입니다.

## 5. Class 2 중심 실험

global 실험 결과를 먼저 본 후:

```bash
CUDA_VISIBLE_DEVICES=1 python experiment.py \
  --seeds 0 1 2 3 4 \
  --epochs 25 \
  --batch-size 2048 \
  --target-class 2 \
  --out-dir runs/class2_5seeds
```

score는 class 2 calibration data로 만들지만, 평가는 full 10-class test set에서 수행합니다.

## 6. 주 실험 질문

### Q1. 같은 class가 비슷한 hidden node 집합을 사용하는가?

**중요:** 주 분석에서는 label을 사용하여 node score를 만들지 않습니다.

- `|activation|`
- `|activation| × outgoing-column L2 norm`

으로 각 sample의 top-k node를 정의한 뒤,
label은 same-class / different-class pair를 나누는 용도로만 사용합니다.

이렇게 해야 "같은 class라서 같은 output weight를 사용했기 때문에 유사해졌다"는 순환적 설명을 피할 수 있습니다.

결과:
- `path_similarity.csv`
- `aggregate_similarity.csv`
- `class_topk_frequency_*.png`

### Q2. 어떤 score가 실제 node 중요도를 잘 예측하는가?

비교:
- weight
- activation
- weight × activation
- gradient × activation
- exact single-node ablation

결과:
- `criterion_correlations.csv`
- `aggregate_correlations.csv`
- 각 seed의 `node_scores.csv`

### Q3. 실제 dense matrix를 줄여도 성능이 유지되는가?

128-node h2에서 node를 제거할 때:

- `fc2`: 대응 행 제거
- `fc3`: 대응 열 제거

즉 실제로 `256 -> 128 -> 10`이 `256 -> k -> 10`으로 바뀝니다.

비교:
- random
- weight
- activation
- weight × activation
- gradient × activation
- ablation oracle-like baseline

결과:
- `all_pruning_results.csv`
- `aggregate_pruning.csv`
- `pruning_accuracy.png`

## 7. 해석 순서

처음에는 아래 3개만 확인하면 됩니다.

1. `aggregate_similarity.csv`
   - `same_mean > different_mean`인가?
   - 5 seeds에서 gap이 일관적인가?

2. `aggregate_correlations.csv`
   - 어떤 score가 actual ablation과 가장 높은 rank correlation을 갖는가?

3. `aggregate_pruning.csv`
   - 같은 pruning ratio에서 어떤 방법이 accuracy를 가장 잘 보존하는가?

그 다음에 class 2 중심 실험이나 새로운 importance criterion을 설계합니다.

## 8. 아직 하지 않는 것

첫 결과가 나오기 전에는 의도적으로 제외합니다.

- 여러 hidden layer 동시 pruning
- pruning 후 fine-tuning
- CNN/Transformer 확장
- dynamic routing
- 복잡한 신규 score
- GPU 0 사용
- 2-GPU DDP
- latency를 주 결과로 해석

MNIST MLP는 너무 작아서 실제 latency는 CUDA kernel launch overhead의 영향을 많이 받습니다.
MNIST 단계에서는 **accuracy / parameter / MAC**을 먼저 봅니다.
