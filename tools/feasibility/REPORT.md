# DistillDrive R50 (stage2 distribution) 가지치기·양자화 실현 가능성 — RTX 3090 1장

- 대상: `projects/configs/stage2/distilldrive_stage2_distribution.py` + `checkpoint/distilldrive_stage2_distribution.pth`
- 원칙: 재학습 없음. 기존 config·체크포인트·학습 코드는 수정하지 않고, 바꿔야 하는 부분(teacher 제거, 정밀도 모드, fp32 fallback)은 스크립트 안에서 메모리상으로만 적용함
- 배포 모델: `distillation_head` 제외 (`with_distillation=False`; teacher 출력은 KD loss에서만 쓰이므로 추론 결과에는 영향 없음). 체크포인트에서 빠진 키는 `head.distillation_head.*` 260개뿐이고 missing key는 0개
- 측정: 배치 1, 연속된 val 프레임으로 워밍업 10회 후 100회 평균. forward 앞뒤로 `torch.cuda.synchronize()`, H2D 복사는 시간에서 제외. `max_memory_allocated`/`max_memory_reserved` 둘 다 기록
- GPU: RTX 3090, 24,090 MB / torch 1.13.0+cu116 / flash-attn 2.3.2

> **비교 기준 (2026-10-22 확정)**: 이후 모든 지연·메모리 비교는 이 폴더의 측정 스크립트(`infer_bench.py`) 값을 기준으로 함 — **fp32 99.0 ms / peak allocated 1245.1 MB**, **repo 모드 90.8 ms / 1301.8 MB**. 처음 받은 기준 지연 **110.6 ms는 이전 측정 방법으로 잰 값**이고 이 스크립트로는 재현되지 않으므로 비교에 쓰지 않음 (peak 1245 MB는 동일하게 재현됨). 정확도 기준은 기존 값(NDS 0.3803, L2 0.7130, obj_box_col 0.259%) 유지.

## 0. 정밀도 모드 정의 (기준값 해석에 중요)

| 모드 | 내용 |
|---|---|
| `fp32` | `wrap_fp16_model` 없음. 전부 fp32. 단 flash-attn은 원래 코드에서도 항상 fp16 |
| `repo` | `tools/test.py`와 같은 방식. config에 `fp16=`가 있어서 `wrap_fp16_model`이 호출되고, `extract_feat`(backbone/neck)의 `@auto_fp16`이 켜져 **backbone·neck만 fp16**, head는 fp32 |
| `full_amp_raw` | `repo` + head 전체를 `torch.autocast(fp16)`로 실행 |
| `full_amp` | `full_amp_raw`에서 오버플로가 나는 두 곳만 fp32로 되돌린 것: `anchor_projection`, `anchor_encoder` (3장 참고) |

- 기준 지표(NDS 0.3803, L2 0.713 등)는 `tools/test.py`로 낸 값이라 실제로는 **`repo` 모드(부분 fp16)** 결과입니다.
- 기준 메모리(Peak Allocated 1245MB)는 `fp32` 모드 측정값 1245.1MB와 정확히 일치합니다.
- `DistillDrive.forward`의 `@force_fp32`가 바깥 autocast를 꺼 버리기 때문에, 모델 전체를 autocast로 감싸도 head는 fp32로 돕니다. 그래서 head 단에서 autocast를 다시 켜야 합니다(`common.apply_precision`).

---

## 실험 1. 학습 메모리 분해 — **판정: 가능** (단, QAT는 부분 가능)

조건: 실제 stage2 학습과 같게 맞춤. train pipeline + GroupInBatchSampler, 배치 4, `wrap_fp16_model` + GradScaler(32), AdamW, `cumulative_iters=12`, grad clip 25, `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`, frozen teacher 포함. 100 iter(옵티마이저 step 8회) 동안의 최대값입니다.

| 항목 | MB | 비고 |
|---|---:|---|
| 가중치: student (학습 대상) | 329.3 | 86.32M params |
| 가중치: teacher (frozen) | 43.6 | 11.44M params |
| 가중치: 기타 frozen + buffer | 1.1 | |
| **가중치 합** | **374.0** | 실측 할당 377.5 |
| **기울기** | **329.3** | `zero_grad(set_to_none=False)`라 계속 상주 |
| **옵티마이저 상태 (AdamW m, v)** | **658.6** | 첫 step 이후 상주 |
| 입력 배치 (GPU) | 51.6 | |
| **활성값 (iter별 peak − 상주분 − 입력)** | **9,071.7** (평균 9,038.4) | 샘플당 약 2.27GB. 100 iter 내내 변동 ±0.2% |
| 상주분 (step 이후) | 1,484 | 위 세 항목 + 약 119MB (시간적 캐시 등) |
| **Peak allocated** | **10,558.3** | iter 56 |
| **Peak reserved** | **14,526** | |
| 프로세스 전체 (nvidia-smi, CUDA context 포함) | 16,554 | 24,090MB 중 → **여유 약 7.5GB** |

판정과 이유
- **가지치기 후 미세조정: 가능.** 현재 stage2 KD 학습이 배치 4에서 16.6GB/24GB로 돌아갑니다. 구조적 가지치기는 활성값을 줄이는 방향이고, 마스크 기반 가지치기도 마스크 텐서(파라미터 크기 수준, 수백 MB 이하) 정도만 추가됩니다.
- **QAT: 부분 가능 (추정).** fake-quant 비용은 이번에 측정하지 않았습니다. 양자화 대상 층의 입력·출력 사본이 backward용으로 저장되므로, 활성값(9.07GB)이 최악의 경우 약 2배로 늘면 peak allocated가 약 19.6GB가 되고 reserved 오버헤드까지 더하면 24GB를 넘을 수 있습니다. 활성값이 배치에 비례한다는 점(샘플당 약 2.27GB)을 이용해 배치 2 + `cumulative_iters=24`(유효 배치 48 유지)로 줄이면 들어갈 것으로 예상합니다. 실측이 필요합니다.

---

## 실험 2. 모듈별 추론 활성값 메모리 (배포 모델, 배치 1) — **판정: 가능**

transient = 해당 모듈 호출 중 peak − 호출 직전 할당, retained = 호출 후 남는 출력. 100프레임 평균(모든 모듈에서 max와 평균의 차이 0.1MB 이내).

| 모듈 | 파라미터 (M / MB) | fp32 transient / retained | repo transient / retained | full_amp transient / retained |
|---|---|---:|---:|---:|
| img_backbone (R50) | 23.51 / 89.7 | 214.5 / 123.8 | 121.3 / 110.3 | 121.3 / 110.3 |
| img_neck (FPN) | 3.34 / 12.8 | 378.4 / 87.9 | 148.4 / 52.6 | 148.4 / 52.6 |
| feature_maps_format | – | 175.3 / 87.7 | 87.7 / 43.8 | 87.7 / 43.8 |
| det_head | 21.30 / 81.3 | 25.0 / 2.2 | 26.4 / 2.2 | 43.7 / 27.1 |
| map_head | 27.52 / 105.0 | 45.9 / 0.3 | 46.7 / 0.3 | 91.5 / 48.4 |
| **motion_plan_head** | 10.88 / 41.5 | **799.7** / 2.6 | **800.6** / 2.6 | **800.1** / 11.3 |
| post_process | – | 7.7 / 0.0 | 7.8 / 0.0 | 7.8 / 0.0 |
| 배포 모델 전체 | 86.55 / 330.2 | | | |

핵심 발견: **motion_plan_head의 약 800MB가 추론 활성값 peak를 결정합니다** (정밀도와 무관).
- profiler(`probe_motion_head_fp32.txt`)로 확인한 원인은 `InstanceQueue.prepare_motion`([instance_queue.py:130-139](../../projects/mmdet3d_plugin/models/motion/instance_queue.py#L130-L139))입니다. 추적 매칭을 `(match[..., None] * temp_feature[:, None]).sum(dim=2)`로 구현해서, `[1,900,900,1] × [1,1,900,256]` = **829MB fp32 텐서**를 큐 길이(4)만큼 매 프레임 만들고 버립니다. 이 `aten::mul` 하나가 전체 CUDA 시간의 약 15%(12.5ms)를 차지합니다.
- 이 연산은 one-hot 행렬곱과 같습니다. `torch.matmul(match.float(), temp_feature)`로 바꿔 검증한 결과는 다음과 같습니다(`probe_queue_matmul.py`, 소스 미수정·스크립트 내 교체).

| 모드 | 원본 지연 / peak alloc / reserved | matmul 지연 / peak alloc / reserved | 연산 단위 최대 차이 |
|---|---|---|---|
| fp32 | 99.4ms / 1245.1 / 1864 | **88.8ms / 857.3 / 1074** | **0.0** (860회 호출, 행당 매칭 ≤ 1) |
| full_amp | 92.4ms / 1374.1 / 1638 | **81.8ms / 620.1 / 846** | **0.0** (matmul만 autocast 끄고 fp32) |

  같은 입력에 대해 결과가 비트 단위로 같으므로 재학습 없이 적용 가능한 무손실 개선입니다. 전체 검증셋 검증은 아래 "실험 A"에 있습니다.
- 가지치기 관점: backbone·neck 채널을 줄여도 줄어드는 추론 transient는 수백 MB 수준입니다. 추론 peak 메모리를 가장 크게 줄이는 방법은 가지치기가 아니라 위 연산 교체입니다.
- 판정: 배포 모델 추론 peak는 1.2~1.4GB(reserved 1.6~1.9GB)라 3090에서 실행하는 데 전혀 문제가 없습니다.

---

## 실험 3. FP16 (torch.autocast) 추론 — **판정: 부분 가능**

### 3-1. 속도·메모리 (배치 1, 워밍업 10 + 100회)

| 모드 | 지연 평균 (ms) | p90 | FPS | Peak allocated (MB) | Peak reserved (MB) |
|---|---:|---:|---:|---:|---:|
| fp32 | 99.0 ± 0.4 | 99.6 | 10.10 | 1245.1 | 1864 |
| repo (backbone·neck fp16) | **90.8** ± 0.3 | 91.1 | **11.03** | 1301.8 | **1586** |
| full_amp_raw (fallback 없음) | 93.3 ± 0.3 | 93.8 | 10.72 | 1374.4 | 1638 |
| **full_amp** (head까지 fp16 + fallback) | 92.1 ± 0.5 | 92.6 | 10.86 | 1374.1 | 1638 |

- 기준 지연 110.6ms는 이전 측정 방법의 값이고, 이 스크립트의 fp32 측정은 99.0ms입니다. peak 메모리는 정확히 일치하므로 측정 프로토콜 차이(H2D·scatter 포함 여부 등)로 보입니다. 이후 비교 기준은 이 표의 fp32 99.0ms / repo 90.8ms입니다.
- head까지 fp16으로 돌리면 backbone만 fp16인 경우(repo)보다 **1.3ms 느리고, peak allocated는 72MB 늘어납니다.** head는 작은 텐서 위주라 커널 실행 오버헤드가 지배적이고, autocast가 매 프레임 가중치를 fp16으로 다시 캐스팅해 그 사본을 들고 있기 때문입니다(실험 2에서 det/map head의 retained 증가로 확인).

### 3-2. 전체 검증셋 평가 (6,019 프레임, 배포 모델)

| 모드 | L2 (m) | obj_box_col | NDS | Det mAP | Map mAP |
|---|---:|---:|---:|---:|---:|
| 기준 (사용자 제공, repo 모드, `--deterministic`) | 0.7130 | 0.259% | 0.3803 | 0.2990 | 0.4207 |
| fp32 (이번) | 0.7118 | 0.252% | 0.3806 | 0.2984 | 0.4206 |
| repo (이번, 기준과 같은 모드) | 0.7137 | 0.254% | 0.3807 | 0.2990 | 0.4209 |
| **full_amp (이번)** | **0.7137** | **0.250%** | **0.3806** | **0.2991** | **0.4211** |
| full_amp_raw | 평가 불가 (436번째 프레임부터 NaN) | | | | |

- 같은 repo 모드인데도 기준과 이번 실행이 L2 0.0007, obj_box_col 0.005%p, NDS 0.0004만큼 다릅니다. 이것이 실행 간 비결정성(flash-attn 등)에 따른 잡음 수준입니다. **full_amp의 차이는 모두 이 범위 안**에 있으므로 정확도 손실은 없다고 판단합니다.
- 커스텀 CUDA 연산 `DeformableAggregation`: 커널이 `float*` 전용이고, 파이썬 래퍼([deformable_aggregation.py:18-22](../../projects/mmdet3d_plugin/ops/deformable_aggregation.py#L18-L22))가 입력을 항상 `.float()`로 바꿉니다. 그래서 **autocast 안에서도 이 연산만 자동으로 fp32로 돌고**, 별도 fallback은 필요 없었습니다(측정 시 입력·출력 모두 fp32로 확인).

### 판정과 이유
- **FP16 추론 자체는 가능**: 2곳만 fp32로 남기면 전체 검증셋에서 NaN 0건이고, 지표도 잡음 범위 안입니다.
- **"부분"인 이유**: (1) 그냥 `torch.autocast`만 씌우면 NaN이 나서 바로는 못 씁니다(아래 막힌 부분 1). (2) 속도 이득은 backbone·neck을 fp16으로 바꾼 데서 다 나오고(−8.3%), head를 fp16으로 바꾸면 오히려 손해입니다. 실용적인 최적점은 `repo` 모드(backbone·neck fp16, head fp32)입니다.

---

## 실험 A. prepare_motion matmul 교체 — 전체 검증셋 검증 (repo 모드) — **판정: 무손실 확인**

`queue_matmul_eval.py`로 원래 코드 / matmul 교체 코드를 각각 전체 검증셋(6,019 프레임)에서 평가하고, 프레임별 원시 출력을 `--out`으로 저장했습니다(`outputs_A/base_run1.pkl`·`base_run2.pkl`·`matmul.pkl`, 생성 정보는 `outputs_A/README.md`). 같은 원래 코드를 한 번 더 돌린 `original_2`가 실행 간 잡음 하한입니다. 세 실행 모두 시드 0, 모드 `repo`.

**설정·전제 확인**

| 항목 | 값 |
|---|---|
| `torch.backends.cuda.matmul.allow_tf32` | **False** (torch 1.13 기본값) → 교체한 matmul은 정확한 fp32 |
| `torch.backends.cudnn.allow_tf32` | True (conv에만 해당, 두 실행에 동일) |
| `cudnn.benchmark` / `cudnn.deterministic` | False / False |
| match 행렬 | 6,018회 호출 모두 `torch.bool`, 값 {0, 1}, **행당·열당 최대 1개** → 부분 순열(one-hot) 행렬. 따라서 matmul은 "골라 오기"와 수학적으로 동일 |

**지표 비교**

| 실행 | L2 (m) | obj_box_col | NDS | Det mAP | Map mAP |
|---|---:|---:|---:|---:|---:|
| 기준 (사용자 제공, 원래 코드) | 0.7130 | 0.259% | 0.3803 | 0.2990 | 0.4207 |
| 원래 코드 — 실험 3의 repo 평가 | 0.7137 | 0.254% | 0.3807 | 0.2990 | 0.4209 |
| 원래 코드 — 이번 실행 | 0.7125 | 0.257% | 0.3810 | 0.2996 | 0.4207 |
| **matmul 교체** | **0.7121** | **0.255%** | **0.3806** | **0.2984** | **0.4210** |

원래 코드 3회의 범위는 L2 0.7125~0.7137, obj_box_col 0.254~0.259%, NDS 0.3803~0.3810, Det mAP 0.2990~0.2996, Map mAP 0.4207~0.4209입니다. matmul은 obj_box_col·NDS가 이 범위 안이고, L2(−0.0004)·Det mAP(−0.0006)·Map mAP(+0.0001)는 범위 끝에서 범위 폭과 비슷한 크기만큼 벗어납니다. 원래 코드끼리의 흔들림과 같은 크기라서 교체에 따른 차이로 볼 근거는 없습니다.

**샘플 단위 출력 비교** (`compare_outputs.py`, 프레임별 최대 |차이|의 분포)

| 출력 | 원래 vs 원래 2회차 (잡음 하한) 중앙값 / p99 / 최대 | 원래 vs matmul 중앙값 / p99 / 최대 |
|---|---|---|
| det 점수 (정렬) | 0.0224 / 0.119 / 0.284 | 0.0221 / 0.126 / 0.281 |
| map 점수 (정렬) | 0.0031 / 0.084 / 0.213 | 0.0031 / 0.076 / 0.159 |
| final_planning (m) | 0.0200 / 1.18 / 5.66 | 0.0191 / 1.15 / 4.16 |
| planning (m) | 0.0398 / 0.444 / 2.25 | 0.0388 / 0.439 / 1.88 |
| planning_score | 0.00032 / 0.248 / 0.527 | 0.00031 / 0.253 / 0.534 |

- 같은 코드끼리도 모든 프레임(6,019/6,019)의 출력이 다르고, 그 크기가 matmul 교체 시의 차이와 분포까지 같습니다. **matmul 교체가 만드는 추가 차이는 이 잡음보다 작아서 구분되지 않습니다.** 연산 단위로는 차이 0.0입니다(실험 2).
- 샘플 단위 "최대 오차"가 0이 될 수 없는 이유는 추론 자체가 결정적이지 않기 때문입니다(자세한 내용과 흔들림 크기는 "평가 비결정성" 절).
  1. `DeformableAggregation` 순전파 커널이 `atomicAdd`로 누적합니다([deformable_aggregation_cuda.cu:183](../../projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu#L183)). 그래서 0번 프레임 det 점수부터 약 5e-5 차이가 납니다.
  2. 평가 중에도 motion_plan_head가 잠재변수를 샘플링합니다(`torch.randn_like(present_mu)`, [motion_planning_head.py:596-597](../../projects/mmdet3d_plugin/models/motion/motion_planning_head.py#L596-L597)). 잡음이 instance 인덱스 순서로 배정되는데, 1번 때문에 top-k 순서가 바뀌면 같은 시드라도 각 객체에 다른 잡음이 붙습니다. 이 때문에 시드를 고정해도 0번 프레임 planning이 약 6e-3 m 다릅니다(잡음을 0으로 두면 약 4e-4 m).
  3. 1·2의 작은 차이가 시간적 캐시(instance bank의 top-k 선택, 추적 ID)를 거치며 누적되고, 점수순으로 정렬된 출력의 순서가 바뀌어 원소 단위 차이가 커 보입니다.
- 참고: `tools/test.py`의 `--deterministic`은 `--seed`를 줄 때만 적용되므로, 기준 평가도 결정적인 실행이 아니었습니다.

**역전파 기울기 비교** (`probe_queue_grad.py`): 실제 학습 상태(train 모드, 학습 데이터, 배치 4, forward + backward) 10 iter에서 `prepare_motion`의 실제 입력을 모았습니다. 9회 호출, 큐 텐서 60개(feature 30, anchor 30)이고, 각각 같은 upstream 기울기 4종(시드 0·1·2 난수, 전부 1)으로 두 식을 backward해 총 240회 비교했습니다.

| 항목 | 결과 |
|---|---|
| temp_feature(큐 feature) 기울기 최대 오차 | **0.0** (기울기 크기 최대 5.24) |
| anchor 큐 기울기 최대 오차 | **0.0** |
| 원래 식·matmul 각각 vs 해석해 matchᵀ·g | 0.0 / 0.0 |
| forward 최대 오차 (같은 입력) | 0.0 |
| match 열당 최대 매칭 수 | 1 (one-hot이라 기울기도 원소 하나를 그대로 옮기는 연산 → 반올림 없음) |

match의 bool→float 변환에서 기울기가 흐르는지:
- 학습 중 실제 상태: match는 `torch.bool`이고 `requires_grad=False`입니다. `match.to(float32)`도 `requires_grad=False`, `grad_fn=None`이며, matmul 식의 autograd 그래프는 match에 닿지 않습니다. 따라서 **match 쪽으로 흐르는 기울기는 없고**, 두 식 모두 큐 텐서로만 기울기가 갑니다.
- 더 나아가 **실제 학습에서는 큐 텐서도 `requires_grad=False`**(`grad_fn=None`, dtype fp32, 호출 시 autocast 꺼짐)입니다. 큐에 넣을 때 `instance_feature.detach()`로 끊기 때문입니다. 그래서 `prepare_motion`은 원래 식이든 matmul이든 학습 기울기에 전혀 관여하지 않습니다. 위의 기울기 비교는 큐 텐서에 기울기를 강제로 켠 가정 상황에서도 같다는 확인입니다.
- 참고 (가정): match를 float로 바꾸고 기울기를 강제로 요구하면, 두 식의 match 기울기는 최대 5.3e-5 차이가 납니다(크기 최대 74.8, 상대 약 7e-7). 256채널 합의 덧셈 순서 차이로 생기는 반올림이고, 실제 코드에서는 match가 기울기를 받지 않으므로 해당이 없습니다.

## 실험 B. 학습 중 모듈별 활성값과 with_cp 비교 (배치 4, 60 iter) — **판정: 가능, with_cp는 켜 두는 것이 유리**

조건은 실험 1과 같습니다(실제 stage2 학습 설정, frozen teacher 포함, `max_split_size_mb:128`). 각 설정마다 hook 실행(모듈별 메모리)과 hook 없는 실행(iter 시간)을 따로 60 iter씩 돌렸습니다. hook은 구간마다 peak 카운터를 리셋하지만 누적 최대값을 따로 유지해서, 두 실행의 peak allocated가 10,558MB로 일치합니다.

**B-1. 설정 확인**

- config `img_backbone.with_cp = True`이고, 실제로 구성된 모델에서도 R50의 모든 Bottleneck 블록이 `with_cp=True`입니다. `frozen_stages=-1`, `norm_eval=False`라 backbone 전체가 학습 대상입니다.

| 모듈 | 학습 대상 (requires_grad) | frozen |
|---|---:|---:|
| img_backbone | 23.508M | 0 |
| img_neck | 3.344M | 0 |
| depth_branch | 0.001M | 0 |
| det_head | 21.073M | 0.231M (instance_feature, 고정 키포인트 스케일) |
| map_head | 27.520M | 0 |
| motion_plan_head | 10.877M | 0.002M |
| distillation_head (teacher) | 0 | 11.437M |
| **합계** | **86.322M** | **11.669M** |

**B-2. 모듈별 학습 활성값** (forward hook. 보유 = forward 후 남아 backward까지 유지되는 양, 일시 = forward 중 최대 증가량. 60 iter 평균, 최대값과의 차이는 모든 모듈에서 약 2% 이내)

| 모듈 | with_cp 켬: 보유 / 일시 (MB) | with_cp 끔: 보유 / 일시 (MB) |
|---|---:|---:|
| img_backbone | **1,406** / 1,432 | **3,649** / 3,654 |
| img_neck | 359 / 577 | 360 / 577 |
| depth_branch (학습 전용) | 178 / 533 | 178 / 533 |
| distillation_head (teacher, no grad) | 2 / 98 | 2 / 97 |
| det_head | 2,114 / 2,122 | 2,106 / 2,114 |
| map_head | 1,559 / 1,636 | 1,562 / 1,639 |
| **motion_plan_head** | 1,061 / **3,220** | 1,059 / **3,220** |
| head.loss | 5 / 28 | 5 / 28 |
| backward 중 최대 증가 | 40 | 40 |

- **학습 peak는 motion_plan_head 안에서 생깁니다.** 그 시점까지 쌓인 보유분(backbone~map_head 약 5.6GB)에 motion_plan_head의 일시 메모리 3.22GB가 더해지는 지점이 peak입니다. 이 일시 메모리는 실험 2에서 찾은 `InstanceQueue.prepare_motion`의 브로드캐스트를 배치 4로 계산한 크기(4×900×900×256×4B = 3,164MiB)와 일치합니다. 아래 B-4에서 matmul로 바꿔 직접 확인했습니다.
- 활성값 보유량 순위(with_cp 켬)는 det_head 2.1GB > map_head 1.6GB > backbone 1.4GB > motion_plan_head 1.1GB > neck 0.36GB입니다. backbone은 checkpointing이 이미 적용돼 있어서, 활성값 기준으로는 head 쪽이 더 큽니다.

**B-3. with_cp 켬/끔 비교** (hook 없는 실행, 앞 10 iter 제외한 50 iter 평균)

| | with_cp 켬 (현재 config) | with_cp 끔 | 차이 |
|---|---:|---:|---:|
| 활성값 최대 | 9,072 MB | 11,308 MB | +2,236 MB |
| peak allocated | 10,558 MB | 12,792 MB | +2,234 MB |
| peak reserved | 14,526 MB | 16,788 MB | +2,262 MB |
| 프로세스 전체 (nvidia-smi) | 16,554 MB | 18,816 MB | +2,262 MB |
| GPU 연산 시간 / iter | 938 ms | 905 ms | −33 ms (−3.5%) |
| 데이터 대기 / iter | 6.0 ms | 6.2 ms | – |

- with_cp를 끄면 3.5% 빨라지는 대신 메모리가 2.2GB 늘고, 24GB 중 여유가 약 7.5GB에서 약 5.3GB로 줄어듭니다. 가지치기·QAT 미세조정에 여유를 남기려면 **켜 두는 것이 유리**합니다.

**B-4. matmul 교체 상태 재측정** (with_cp 켬, 배치 4, 60 iter. 원래 코드 값은 B-3 실행)

| | 원래 식 | matmul | 차이 |
|---|---:|---:|---:|
| motion_plan_head 일시 (hook) | 3,220 MB | 1,124 MB | −2,096 MB |
| 활성값 최대 | 9,072 MB | 6,944 MB | −2,128 MB |
| **peak allocated** | **10,558 MB** | **8,431 MB** | **−2,127 MB (−20%)** |
| peak reserved | 14,526 MB | 11,362 MB | −3,164 MB |
| **프로세스 전체 (nvidia-smi)** | **16,554 MB** | **13,390 MB** | **−3,164 MB** |
| **GPU 연산 시간 / iter** | **938 ms** | **887 ms** | **−51 ms (−5.4%)** |

- 예상(peak 약 −3GB)보다 peak allocated 감소가 작은(−2.1GB) 이유: 브로드캐스트 텐서(3,164MiB)가 사라지면 motion_plan_head 자체 활성값(보유 1,061MB + 일시분)이 새로운 국소 최대가 됩니다. 예상치는 이 몫을 빼지 않았습니다. 반면 peak reserved와 프로세스 전체는 정확히 3,164MB 줄었습니다. 할당기가 그 텐서 크기의 블록을 따로 잡아 두던 것이 없어졌기 때문입니다.
- 다른 모듈의 보유·일시 메모리는 원래 식과 같습니다(차이 ≤ 1MB).
- 참고: iter별 loss는 첫 iter 차이 0.0002, 60 iter 평균 19.471 대 19.460입니다. 이후 iter별로는 최대 0.69(3.4%) 차이가 나지만, 학습도 결정적이지 않아(atomicAdd 역전파 등) 같은 코드끼리의 차이를 재지 않았으므로 교체 영향으로 판단할 수 없습니다.

---

## 평가 비결정성 — 원인과 흔들림 크기

같은 체크포인트·같은 코드·같은 모드(repo)로 평가해도 실행마다 출력과 지표가 달라집니다. 지표 차이를 판단할 때는 아래 흔들림 범위와 비교해야 합니다.

**원인 (코드와 측정으로 확인)**

| 원인 | 위치 | 성질 |
|---|---|---|
| 1. DeformableAggregation 순전파 `atomicAdd` | [deformable_aggregation_cuda.cu:183](../../projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu#L183) | 여러 스레드가 같은 출력에 더하는 순서가 실행마다 달라 fp32 반올림이 달라짐. 시드와 무관 |
| 2. 평가 중 잠재변수 샘플링 `torch.randn_like(present_mu)` | [motion_planning_head.py:596-597](../../projects/mmdet3d_plugin/models/motion/motion_planning_head.py#L596-L597) | 평가 때도 `mu + sigma * noise`를 씀. 잡음이 instance 인덱스 순서로 배정되는데, 1번 때문에 top-k 순서가 바뀌면 같은 시드라도 각 객체에 다른 잡음이 붙음 → **시드 고정으로 통제 안 됨** |
| 3. 시간적 전파에 의한 누적 | instance bank top-k 선택, 추적 ID, InstanceQueue | 1·2의 작은 차이가 다음 프레임의 쿼리 선택·순서를 바꾸며 커짐 |
| 참고 | `tools/test.py` | `--deterministic`은 `--seed`를 줄 때만 적용됨. 기준 평가도 결정적이지 않았음 |

배경 확인: 같은 입력이면 backbone 특징은 완전히 같습니다(차이 0.0). 학습 전용 난수(GridMask, attention dropout)는 평가 때 동작하지 않습니다.

**흔들림 크기 — 단일 프레임 (같은 입력을 한 프로세스에서 반복, 확인)**

| 조건 | det 점수(정렬) | map 점수(정렬) | final_planning |
|---|---:|---:|---:|
| 같은 시드, 샘플링 켬 | 약 5e-5 | 약 1e-4 | 약 5.5e-3 ~ 7e-3 m |
| 잡음을 0으로 고정 (원인 1만 남김) | 약 5e-5 | – | 약 4e-4 m |

**흔들림 크기 — 검증셋 전체, 샘플 단위 (원래 코드 vs 원래 코드 2회차, 6,019프레임, 확인)**

| 출력 | 중앙값 | p99 | 최대 |
|---|---:|---:|---:|
| det 점수 (정렬) | 0.022 | 0.119 | 0.284 |
| map 점수 (정렬) | 0.0031 | 0.084 | 0.213 |
| final_planning (m) | 0.020 | 1.18 | 5.66 |
| planning (m) | 0.040 | 0.444 | 2.25 |

0번 프레임에서는 1e-5~1e-3 수준이던 차이가 시간적 전파를 거치며 30프레임 안에 det 점수 기준 약 5e-2까지 커집니다. 6,019프레임 모두 차이가 있습니다.

**흔들림 크기 — 지표 (repo 모드, 원래 코드 4회: 기준값 + 이번 실행 3회, 확인)**

| 지표 | 최소 ~ 최대 | 범위 폭 |
|---|---|---:|
| L2 (m) | 0.7125 ~ 0.7137 | 0.0012 |
| obj_box_col | 0.254% ~ 0.259% | 0.005%p |
| NDS | 0.3803 ~ 0.3810 | 0.0007 |
| Det mAP | 0.2990 ~ 0.2996 | 0.0006 |
| Map mAP | 0.4207 ~ 0.4209 | 0.0002 |

실행 4회로 잡은 범위라 실제 분포보다 좁을 수 있습니다(추정). 이 폭보다 작은 지표 차이는 변경의 영향으로 판단하지 않습니다.

**줄이는 방법 (미적용, 소스 수정 필요)**
- 원인 2: 평가 때 `noise`를 0으로 두면(`sample = mu`) 샘플링에 의한 흔들림이 없어집니다. 다만 모델 동작이 바뀌므로 기준 지표도 다시 재야 합니다.
- 원인 1: DAF 순전파를 `atomicAdd` 없는 결정적 커널로 바꿔야 합니다.

---

## 실험 5 준비. det_head·map_head 층별·연산별 분해 (가지치기 실행 X)

학습 활성값은 `train_memory.py --hooks heads`로 쟀습니다(배치 4, with_cp 켬, matmul 적용, 60 iter 평균, 보유량 = forward 후 backward까지 남는 양). 추론 시간은 `head_breakdown.py --infer`로 쟀습니다(배포 모델, repo 모드, matmul 적용, 배치 1, 워밍업 10 + 100프레임, 구간마다 CUDA 이벤트 기록). 이벤트를 켠 상태의 프레임 지연은 80.7ms입니다(원래 식 repo 90.8ms에서 matmul로 약 10ms 줄어든 것과 일치).

**구조** (config와 구성된 모델에서 확인)

| | det_head | map_head |
|---|---|---|
| 쿼리 수 (시간적 쿼리) | 900 (600) | 100 (33), 쿼리당 점 20개 |
| decoder 층 수 | 6 | 6 |
| 층 구성 | L0: deformable→ffn→norm→refine / L1~5: temp_gnn→gnn→norm→deformable→ffn→norm→refine | L0: gnn→norm→deformable→ffn→norm→refine / L1~5: temp_gnn 추가 |
| 잔차 스트림 폭 | 256 | 256 |
| FFN | 512 → **1024** → 256 (+ identity_fc 512→256) | 512 → **1024** → 256 |
| attention | 512차원(feature+pos cat, decouple), 8 heads, flash-attn | 256차원, 8 heads, flash-attn |
| deformable | 그룹 8, 레벨 4, 카메라 6 | 같음 |

**연산별 합계 (6개 층 합, 확인된 값)**

| 연산 | det 학습 보유 MB (비율) | det 추론 ms (비율) | map 학습 보유 MB (비율) | map 추론 ms (비율) |
|---|---:|---:|---:|---:|
| self attention | 244 (11.6%) | 1.71 (8.6%) | 15 (0.9%) | 0.74 (3.1%) |
| temporal attention | 243 (11.5%) | 1.65 (8.3%) | 10 (0.6%) | 0.52 (2.2%) |
| **deformable (커스텀 CUDA)** | **775 (36.7%)** | **9.54 (47.8%)** | **1,463 (93.8%)** | **20.57 (86.0%)** |
| FFN | 261 (12.3%) | 1.14 (5.7%) | 31 (2.0%) | 0.56 (2.3%) |
| └ 그중 은닉층 (Linear→1024, ReLU, Dropout) | 191 (9.0%) | 0.48 (2.4%) | 23 (1.5%) | 0.14 (0.6%) |
| └ 그중 출력 Linear (1024→256) | 22 (1.0%) | 0.35 (1.8%) | 2 (0.2%) | 0.27 (1.1%) |
| refine | 326 (15.4%) | 2.14 (10.7%) | 27 (1.7%) | 0.67 (2.8%) |
| anchor encoder | 243 (11.5%) | 2.81 (14.1%) | 12 (0.7%) | 0.31 (1.3%) |
| LayerNorm | 40 (1.9%) | 0.10 (0.5%) | 5 (0.3%) | 0.09 (0.4%) |
| 그 외 (instance bank 등, 총합 − 위 합) | −17 (−0.8%) | 0.86 (4.3%) | −2 (−0.1%) | 0.47 (2.0%) |
| **head 전체** | **2,114** | **19.96** | **1,559** | **23.93** |

- 층별로는 거의 균일합니다. det는 층당 deformable 129MB / 1.6ms, FFN 43MB / 0.19ms이고, map은 층당 deformable 244MB / 3.4ms입니다. 층별 값은 `outputs/exp5prep_head_breakdown.json`의 `per_layer`에 있습니다.
- "그 외"의 작은 음수는 각 구간의 보유량에 head 안에서 나중에 교체·해제되는 출력(예: 층마다 새로 만드는 anchor_embed)이 들어 있어서, 부분 합이 전체보다 조금 커지기 때문입니다.
- **두 head 모두 deformable이 가장 큽니다.** map_head는 쿼리 100개지만 쿼리당 점 20개 × 키포인트 때문에 deformable 하나가 메모리 94%, 시간 86%를 차지합니다.

**FFN 은닉 차원 가지치기 효과 (추정)**

가정: 은닉 차원을 r만큼 자르면 은닉 차원을 가진 텐서(ReLU 출력, dropout 마스크·출력)의 보유 메모리가 r만큼 줄어든다. 시간은 은닉층과 출력 Linear 시간이 r에 비례해 준다고 보는 **상한**이다. 작은 GEMM은 실제로는 이보다 덜 줄어든다.

| 가지치기 | det+map 학습 메모리 감소 | 학습 활성값 대비 | 학습 peak 대비 | 추론 시간 감소 (상한) | 프레임 지연 대비 |
|---|---:|---:|---:|---:|---:|
| 20% | 43 MB | 0.6% | 0.5% | 0.25 ms | 0.3% |
| 40% | 86 MB | 1.2% | 1.0% | 0.50 ms | 0.6% |

→ FFN 가지치기로 줄일 수 있는 메모리·시간은 1% 안팎입니다. 이 모델에서 FFN은 가지치기 대상으로서 효과가 작습니다.

**Torch-Pruning으로 채널을 자를 수 있는 구조인지** (코드 구조 분석에 따른 판단. Torch-Pruning은 설치하지 않았고 의존성 추적도 돌리지 않음 → 추정)

| 연산 | 판단 | 커스텀 CUDA | 근거 |
|---|---|---|---|
| FFN | **가능** | 없음 | Linear→ReLU→Linear 표준 MLP. 은닉 1024만 자르면 잔차 스트림(256)에 영향 없음 |
| refine | 내부 차원만 가능 | 없음 | linear_relu_ln(256) 블록. 입력(256)·출력(box/cls/quality)은 고정 |
| anchor encoder | 내부 차원만 가능 | 없음 | pos/size/yaw/vel MLP. 마지막 출력은 cat되어 query_pos(256)가 되므로 고정 |
| self/temporal attention | 어려움 | **flash-attn 커널** | `FlashMHA`가 `nn.Linear`가 아니라 원시 `in_proj_weight`를 chunk해서 씀 → 기본 추적 불가, 사용자 정의 pruner 필요. head_dim 제약도 있음 |
| deformable | 불가에 가까움 | **DAF (자체 CUDA)** | 커널 입력이 FPN 특징맵 채널(256)·그룹(8)과 묶여 있음 |
| LayerNorm / 잔차 스트림(256) | 불가 | – | det/map/motion head, DAF, FPN 출력, KD 특징에 공통 |

→ 메모리·시간을 가장 많이 쓰는 deformable(det 37%/48%, map 94%/86%)과 attention은 Torch-Pruning으로 바로 자르기 어렵고, 쉽게 자를 수 있는 FFN은 비중이 작습니다. 가지치기 대상을 정할 때 이 구조적 제약을 함께 봐야 합니다.

---

## 층 수 시험 (재학습 없음) — **판정: 재학습 없이는 불가**

**사전 확인 (코드, 확인)**
1. 두 head 모두 층마다 `refine`이 중간 예측을 내고, 학습 시 6개 층 전부에 같은 가중치로 손실이 걸립니다([detection3d_head.py:420](../../projects/mmdet3d_plugin/models/detection3d/detection3d_head.py#L420)의 `decoder_idx` 반복). map_head도 같은 `Sparse4DHead` 클래스입니다. 학습 로그에 `det_loss_*_0`~`_5`, `map_loss_*_0`~`_5`가 모두 있습니다.
2. instance bank: `update`는 첫 층 refine 직후(`num_single_frame_decoder=1`)에 한 번 불리고, `cache`·추적 ID는 마지막 층 출력을 씁니다. motion_plan_head는 `instance_feature`·`anchor_embed`(마지막 층 이후)와 `classification[-1]`·`prediction[-1]`을 받고, det/map/motion 후처리도 `output_idx=-1`입니다. → 층을 뒤에서 자르면 이 모두가 별도 수정 없이 멈춘 층의 출력을 받습니다.
3. deformable 설정: det 쿼리당 점 13개(학습형 6 + 고정 7), map 쿼리당 점 300개, 둘 다 카메라 6 / 특징맵 단계 4 / 그룹 8(그룹당 32채널). 샘플링 점 합계 det 11,700 · map 30,000.

**방법**: `tools/feasibility/configs/`의 새 설정(기존 config를 `_base_`로 상속, `operation_order`만 앞에서 자름. 나머지 설정이 기존과 같음을 확인). 남은 층의 가중치는 체크포인트와 같고, 잘라낸 층의 키만 무시합니다. 모든 측정은 matmul 적용 상태, 평가는 repo 모드, 검증셋 전체입니다.

**학습 메모리·속도 (배치 4, 60 iter, with_cp 켬, 확인)**

| 층 수 (det·map) | peak allocated | 프로세스 전체 | iter 연산 시간 |
|---|---:|---:|---:|
| 6 (기준) | 8,431 MB | 13,390 MB | 887 ms |
| 5 | 7,674 MB (−757) | 12,634 MB | 790 ms (−11%) |
| 4 | 6,926 MB (−1,505) | 11,902 MB | 690 ms (−22%) |

**평가 지표와 추론 지연 (확인)** — 괄호는 6층 대비 차이. "잡음 내"는 평가 흔들림 범위 폭(L2 0.0012, obj_box_col 0.005%p, NDS 0.0007, Det mAP 0.0006, Map mAP 0.0002; "평가 비결정성" 절) 이하라는 뜻

| 설정 | L2 (m) | obj_box_col | NDS | Det mAP | Map mAP | 추론 지연 / FPS |
|---|---|---|---|---|---|---|
| 6층 (기준) | 0.7121 | 0.255% | 0.3806 | 0.2984 | 0.4210 | 79.9 ms / 12.5 |
| 5층 | 1.3482 (+0.636) | 0.590% (+0.335%p) | 0.2928 (−0.088) | 0.1702 (−0.128) | 0.3202 (−0.101) | 72.8 ms / 13.7 |
| 4층 | 1.2835 (+0.571) | 0.582% (+0.327%p) | 0.2497 (−0.131) | 0.1077 (−0.191) | 0.2824 (−0.139) | 65.8 ms / 15.2 |
| det만 4층 | 0.7466 (+0.035) | 0.346% (+0.091%p) | 0.2493 (−0.131) | 0.1080 (−0.190) | 0.4209 (−0.0001, 잡음 내) | 73.6 ms / 13.6 |
| map만 4층 | 1.0663 (+0.354) | 0.317% (+0.062%p) | 0.3803 (−0.0002, 잡음 내) | 0.2984 (0.0000, 잡음 내) | 0.2821 (−0.139) | 72.2 ms / 13.9 |

- 잘린 head의 지표만 떨어지고, 안 잘린 head의 지표는 흔들림 범위 안입니다(det만 4층 → Map mAP 그대로, map만 4층 → NDS·Det mAP 그대로).
- **planning(L2)은 map 절단에 훨씬 민감합니다.** map만 4층일 때 +0.354이고 det만 4층일 때는 +0.035입니다. 둘 다 자르면 +0.57~0.64입니다.
- 5층이 4층보다 L2가 나쁩니다(1.348 대 1.284). 층 수에 따라 단조롭게 변하지 않습니다.
- 모든 하락이 흔들림 범위의 수백 배라서 실행 간 잡음으로는 설명되지 않습니다.

**원인 진단 (`probe_layer_exit.py`, 처음 80프레임, 확인)**

| 비교 | 장면 첫 프레임 det / map logit 차이 | 이후 캐시 특징 평균 차이 (det) |
|---|---|---|
| 6층 대 6층 (잡음 하한) | 2.3e-3 / 4.3e-3 | 0.02~0.08 |
| 6층의 5번째 층 대 5층 모델 | 1.3e-3 / 6.7e-3 | 약 1.1 |
| 6층의 4번째 층 대 4층 모델 | 1.1e-3 / 6.3e-3 | 약 1.15 |

- 캐시가 없는 장면 첫 프레임에서는 잘린 모델이 6층 모델의 해당 층 출력과 잡음 수준으로 같습니다 → 층 자르기 구현은 정확합니다.
- 하락 원인은 하류 모듈의 입력 분포가 바뀌는 데 있습니다. 4·5층 출력 특징은 6층 출력과 평균 약 1.1 차이가 나는데(같은 모델끼리의 15~50배), 이 특징이 (1) instance bank 캐시를 거쳐 다음 프레임의 temporal attention 키로, (2) motion_plan_head 입력으로 들어갑니다. 두 모듈은 6층 출력으로만 학습됐습니다. 층별 손실은 박스·클래스 예측만 감독하고, 중간 층 특징이 하류 입력으로 쓰이도록 학습되지는 않았습니다.
- 장면 전환 프레임에서도 차이가 큰 이유: 이전 장면 캐시는 `update`에서 마스크로 버려지지만, temporal attention의 키로는 여전히 쓰입니다.

**학습 로그의 층별 손실 (stage2 distribution, 마지막 iter 70,278, 확인)** — 괄호는 마지막 20개 로그(iter 69,309~70,278) 평균. 배치 4라 단일 값은 잡음이 큼

| 손실 | 층 0 | 층 1 | 층 2 | 층 3 | 층 4 | 층 5 |
|---|---|---|---|---|---|---|
| det cls | 0.538 (0.518) | 0.532 (0.500) | 0.509 (0.482) | 0.492 (0.466) | 0.482 (0.461) | 0.479 (0.458) |
| det box | 1.073 (1.033) | 0.830 (0.795) | 0.785 (0.746) | 0.773 (0.736) | 0.764 (0.727) | 0.763 (0.723) |
| det cns | 0.654 (0.642) | 0.642 (0.632) | 0.631 (0.621) | 0.630 (0.620) | 0.629 (0.619) | 0.630 (0.620) |
| det yns | 0.066 (0.080) | 0.053 (0.066) | 0.050 (0.063) | 0.049 (0.062) | 0.047 (0.060) | 0.048 (0.059) |
| map cls | 0.189 (0.155) | 0.164 (0.131) | 0.151 (0.116) | 0.133 (0.102) | 0.125 (0.096) | 0.128 (0.097) |
| map line | 0.426 (0.373) | 0.271 (0.220) | 0.249 (0.206) | 0.236 (0.200) | 0.236 (0.199) | 0.235 (0.199) |

- 학습 중(6층 출력이 캐시·motion으로 가는 상태)에는 층 3~4 이후 손실이 거의 줄지 않습니다. 중간 층의 예측 능력 자체는 6층에 가깝다는 뜻입니다.
- **추정**: 4·5층으로 자른 구조로 미세조정(재학습)하면 하류 모듈이 바뀐 입력에 적응해 상당 부분 회복될 가능성이 있습니다. 이번 범위(재학습 없음)에서는 확인하지 않았습니다.

**판정**: 층 수를 줄이면 학습 메모리 −0.75GB/층, iter 시간 −11%/층, 추론 지연 −7ms/층이지만, 재학습 없이는 지표가 크게 떨어져 쓸 수 없습니다. 층 수 축소는 미세조정과 함께 검토해야 하는 후보입니다.

---

## 실험 7. 언어 모듈(LLM) 동시 실행 — **보류 - 언어 모듈 모델 미정**

언어 모듈로 쓸 모델이 확정되면 DistillDrive와 동시에 올렸을 때의 메모리 합계와 2토큰/40토큰 생성 시간을 측정합니다.

---

## 막힌 부분

1. **head 전체를 autocast하면 NaN이 발생 → 해결함 (fp32 fallback 2곳).** 로더 순서로 436번째 프레임은 장면 전환이라 Δt = −431,388초입니다. `InstanceBank.get`이 캐시된 앵커를 `anchor += vel × Δt`로 외삽하는데(|vel| ≤ 19.2m/s → 약 8.3×10⁶m), autocast 안의 matmul이 fp16 최대값 65,504를 넘어 inf가 됩니다. fp32에서는 유한한 값이 되고 `mask`로 버려지지만, fp16에서는 inf 앵커가 temporal attention에 들어가 NaN이 되고, NaN 특징이 다시 캐시되어 이후 5,583프레임 전부로 퍼집니다. `anchor_projection`만 fp32로 바꾸면 그다음 `anchor_encoder`의 첫 Linear(fp16)가 같은 이유로 오버플로하므로, 이 둘을 fp32로 고정했습니다(LayerNorm 이후는 값 범위가 정상). 근거: `probe_fp16_nan_full_amp_raw.txt`, `probe_fp16_nan_full_amp.txt`(1,000프레임 NaN 0건), full_amp 전체 평가 통과.
2. **`post_process`에 fp16 텐서가 들어가 오류 → 해결함.** `@force_fp32(apply_to=("model_outs"))`의 인자 이름이 실제 인자와 맞지 않아 캐스팅이 일어나지 않고, decoder의 index_put에서 dtype 불일치가 납니다. 스크립트에서 head 출력을 fp32로 되돌려 해결했습니다.
3. **INT8 양자화는 측정 불가 (막힘).** venv에 TensorRT / torch-tensorrt / pytorch-quantization / ONNX / onnxruntime이 하나도 없고, torch 내장 양자화 엔진(fbgemm, qnnpack, onednn, x86)은 CPU 전용입니다. 또 `DeformableAggregation`은 fp32 커널만 있고 flash-attn은 fp16 전용이라, INT8로 가려면 이 두 연산은 별도 처리(fp16/fp32 유지 또는 INT8 커널 작성)가 필요합니다.
4. **가지치기·QAT 학습의 추가 메모리는 미측정.** 구현이 있어야 측정할 수 있어서, 실험 1의 QAT 판정은 추정치입니다.
5. **기준 지연 110.6ms는 재현하지 못함** (이번 fp32 99.0ms). 이전 측정 방법의 값으로 보고, 비교 기준을 이 스크립트 값(fp32 99.0ms, repo 90.8ms)으로 바꿨습니다. peak 메모리 1245MB는 정확히 재현했습니다.

---

## 전체 결론

| 항목 | 판정 | 근거 요약 |
|---|---|---|
| 학습: 가지치기 후 KD 미세조정 (배치 4) | **가능** | 원래 식: peak alloc 10.6GB / 프로세스 16.6GB (여유 약 7.5GB). matmul 교체 시 peak 8.4GB / 프로세스 13.4GB (여유 약 10.7GB) |
| 학습: QAT (배치 4) | **부분 가능** | fake-quant 오버헤드 미측정. matmul 교체로 여유가 늘어 가능성은 높아짐(추정) |
| 실행: 배포 모델 추론 | **가능** | peak 1.25~1.37GB, 10~11 FPS (matmul 교체 시 peak 0.62GB, 12.5 FPS) |
| 양자화: FP16 | **부분 가능** | fp32 fallback 2곳이 필요하고, 정확도 손실은 없음. 속도 이득은 backbone fp16에서만 나옴 (repo 모드가 최적) |
| 양자화: INT8 | **불가능 (현 환경)** | 툴체인 없음, 커스텀 연산(DAF)과 flash-attn에 INT8 경로 없음 |
| `prepare_motion` matmul 교체 | **무손실 확인** | 연산·기울기 모두 차이 0.0, 검증셋 지표는 흔들림 범위 안. 추론 −10.6ms, 학습 peak −2.1GB·iter −5.4% (실험 A, B-4) |
| FFN 은닉 차원 가지치기 | **가능하나 효과 작음 (추정)** | 40%를 잘라도 학습 메모리 −1.0%, 추론 지연 −0.6% 이하 (실험 5 준비) |
| decoder 층 수 축소 (재학습 없음) | **불가** | 4층: 추론 −14ms, 학습 peak −1.5GB이지만 L2 +0.57, NDS −0.13. 원인은 하류 모듈(캐시·motion)이 6층 출력으로만 학습된 것 (층 수 시험) |

- 3090 한 장에서 **가지치기 + 미세조정 학습과 FP16 추론은 가능**합니다. QAT는 실측이 필요하고, INT8 배포는 툴체인 설치와 커스텀 연산 처리가 먼저 해결돼야 합니다.
- 지금까지 찾은 가장 큰 무손실 이득은 가지치기도 양자화도 아닌 **`InstanceQueue.prepare_motion` 연산 교체**입니다. 적용하려면 `instance_queue.py`를 수정해야 해서 소스에는 반영하지 않았습니다.
- 메모리·시간을 가장 많이 쓰는 연산은 deformable(det 37%/48%, map 94%/86%)과 attention인데, 둘 다 커스텀 CUDA(DAF, flash-attn)라 Torch-Pruning으로 바로 자르기 어렵습니다. 쉽게 자를 수 있는 FFN은 비중이 작습니다.
- 층 수 축소는 지연·메모리 이득이 크지만, 재학습 없이는 지표가 무너집니다. 학습 로그상 층 3~4 이후 손실이 거의 줄지 않으므로, 미세조정과 함께라면 검토할 가치가 있는 후보입니다(추정).
- 지표 비교 시에는 평가 흔들림(L2 ±약 0.0012 폭 등, "평가 비결정성" 절)을 기준으로 판단해야 합니다.

---

## 재현

```bash
cd /workspace/yujung/DistillDrive
source .venv/bin/activate && export PYTHONPATH=$(pwd)
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tools/feasibility/train_memory.py --iters 100   # 실험 1 (~5분)
bash tools/feasibility/run_infer_all.sh   # 실험 2·3 + 진단 (~2시간, 전체 평가 3회 포함)
```

| 파일 | 역할 |
|---|---|
| `common.py` | 설정 로드, 배포 모델 구성(teacher 제외), 정밀도 모드·fp32 fallback |
| `train_memory.py` | 실험 1 → `outputs/exp1_train_memory_bs4.json` (iter별 기록 포함) |
| `module_memory.py` | 실험 2 → `outputs/exp2_module_memory_{mode}.json` |
| `infer_bench.py` | 실험 3 → `outputs/exp3_infer_{mode}.json`, `outputs/exp3_infer_{mode}_eval.json`, `outputs/eval_{mode}/` (전체 지표, 평가 로그) |
| `probe_motion_head.py` | motion_plan_head 800MB의 원인 연산 프로파일 → `outputs/probe_motion_head_fp32.txt` |
| `probe_fp16_nan.py` | FP16 NaN 발생 위치·원인 → `outputs/probe_fp16_nan_{mode}.txt` |
| `probe_queue_matmul.py` | 큐 연산 matmul 교체의 효과·동등성 → `outputs/probe_queue_matmul_{mode}.json` |
| `run_infer_all.sh` | 실험 2·3과 진단을 순서대로 실행 |
| `queue_matmul_eval.py` | 실험 A: 원래/matmul 변형으로 전체 검증셋 평가 + 출력 저장 → `outputs/queue_eval/<run>/` |
| `probe_queue_grad.py` | 실험 A: 실제 학습 입력으로 두 식의 역전파 기울기 비교 → `outputs/probe_queue_grad.json` |
| `compare_outputs.py` | 실험 A: 두 출력 pkl의 샘플 단위 비교 → `outputs/queue_eval/compare_*.json` |
| `run_queue_eval.sh` | 실험 A 전체(원래, matmul, 원래 2회차, 비교) 순차 실행 (약 2시간) |
| `train_memory.py --with-cp on/off [--hooks]` | 실험 B → `outputs/expB_train_bs4_cp{on,off}_{hooks,timing}.json` |
| `run_train_B.sh` | 실험 B 4회 실행 (약 25분) |
| `train_memory.py --queue matmul [--hooks heads]` | B-4 matmul 재측정, 실험 5 준비의 head 분해 학습 메모리 → `outputs/expB_train_bs4_cpon_{timing,hooks,heads}_matmul.json` |
| `configs/stage2_distribution_{layers4,layers5,det4_map6,det6_map4}.py` | 층 수 시험 설정 (기존 config 상속, operation_order만 자름) |
| `run_layers.sh`, `run_layers_combo.sh` | 층 수 시험: 학습 메모리·추론 지연·전체 평가 (각 약 1시간 50분 / 1시간 45분) |
| `probe_layer_exit.py --layers N` | 층 수 시험 진단: 잘린 모델 vs 6층 모델의 해당 층 출력 비교 (N=6이면 잡음 하한) → `outputs/probe_layer_exit_L*.json` |
| `layer_losses.py` | 학습 로그의 층별 손실 표 → `outputs/stage2_layer_losses.json` |
| `head_breakdown.py --infer / --summarize` | 실험 5 준비: head 층·연산별 추론 시간 측정, 학습 메모리와 합쳐 요약·FFN 가지치기 추정 → `outputs/exp5prep_*.json` |
