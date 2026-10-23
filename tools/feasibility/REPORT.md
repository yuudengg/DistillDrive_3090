# DistillDrive R50 (stage2 distribution) 가지치기·양자화 실현 가능성 — RTX 3090 1장

- 대상: `projects/configs/stage2/distilldrive_stage2_distribution.py` + `checkpoint/distilldrive_stage2_distribution.pth`
- 원칙: 재학습 없음. 기존 config·체크포인트·학습 코드는 수정하지 않고, 바꿔야 하는 부분(teacher 제거, 정밀도 모드, fp32 fallback)은 스크립트 안에서 메모리상으로만 적용함
- 배포 모델: `distillation_head` 제외 (`with_distillation=False`; teacher 출력은 KD loss에서만 쓰이므로 추론 결과에는 영향 없음). 체크포인트에서 빠진 키는 `head.distillation_head.*` 260개뿐이고 missing key는 0개
- 측정: 배치 1, 연속된 val 프레임으로 워밍업 10회 후 100회 평균. forward 앞뒤로 `torch.cuda.synchronize()`, H2D 복사는 시간에서 제외. `max_memory_allocated`/`max_memory_reserved` 둘 다 기록
- GPU: RTX 3090, 24,090 MB / torch 1.13.0+cu116 / flash-attn 2.3.2


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

  같은 입력에 대해 결과가 비트 단위로 같으므로 재학습 없이 적용 가능한 무손실 개선입니다. 다만 이 변형으로 전체 검증셋 평가는 돌리지 않았습니다.
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

- 기준 지연 110.6ms와 이번 fp32 측정 99.0ms는 다릅니다. peak 메모리는 정확히 일치하므로 측정 프로토콜 차이(H2D·scatter 포함 여부 등)로 보입니다. 아래 비교는 같은 스크립트·같은 프레임으로 측정한 상대값입니다.
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

## 막힌 부분

1. **head 전체를 autocast하면 NaN이 발생 → 해결함 (fp32 fallback 2곳).** 로더 순서로 436번째 프레임은 장면 전환이라 Δt = −431,388초입니다. `InstanceBank.get`이 캐시된 앵커를 `anchor += vel × Δt`로 외삽하는데(|vel| ≤ 19.2m/s → 약 8.3×10⁶m), autocast 안의 matmul이 fp16 최대값 65,504를 넘어 inf가 됩니다. fp32에서는 유한한 값이 되고 `mask`로 버려지지만, fp16에서는 inf 앵커가 temporal attention에 들어가 NaN이 되고, NaN 특징이 다시 캐시되어 이후 5,583프레임 전부로 퍼집니다. `anchor_projection`만 fp32로 바꾸면 그다음 `anchor_encoder`의 첫 Linear(fp16)가 같은 이유로 오버플로하므로, 이 둘을 fp32로 고정했습니다(LayerNorm 이후는 값 범위가 정상). 근거: `probe_fp16_nan_full_amp_raw.txt`, `probe_fp16_nan_full_amp.txt`(1,000프레임 NaN 0건), full_amp 전체 평가 통과.
2. **`post_process`에 fp16 텐서가 들어가 오류 → 해결함.** `@force_fp32(apply_to=("model_outs"))`의 인자 이름이 실제 인자와 맞지 않아 캐스팅이 일어나지 않고, decoder의 index_put에서 dtype 불일치가 납니다. 스크립트에서 head 출력을 fp32로 되돌려 해결했습니다.
3. **INT8 양자화는 측정 불가 (막힘).** venv에 TensorRT / torch-tensorrt / pytorch-quantization / ONNX / onnxruntime이 하나도 없고, torch 내장 양자화 엔진(fbgemm, qnnpack, onednn, x86)은 CPU 전용입니다. 또 `DeformableAggregation`은 fp32 커널만 있고 flash-attn은 fp16 전용이라, INT8로 가려면 이 두 연산은 별도 처리(fp16/fp32 유지 또는 INT8 커널 작성)가 필요합니다.
4. **가지치기·QAT 학습의 추가 메모리는 미측정.** 구현이 있어야 측정할 수 있어서, 실험 1의 QAT 판정은 추정치입니다.
5. **기준 지연 110.6ms는 재현하지 못함** (이번 fp32 99.0ms). 기준 측정 방식이 확인되지 않았습니다. peak 메모리 1245MB는 정확히 재현했습니다.

---

## 전체 결론

| 항목 | 판정 | 근거 요약 |
|---|---|---|
| 학습: 가지치기 후 KD 미세조정 (배치 4) | **가능** | peak alloc 10.6GB / reserved 14.5GB / 프로세스 16.6GB, 여유 약 7.5GB |
| 학습: QAT (배치 4) | **부분 가능** | fake-quant 오버헤드 미측정. 배치 2 + 누적 24로 낮추면 들어갈 것으로 추정 |
| 실행: 배포 모델 추론 | **가능** | peak 1.25~1.37GB, 10~11 FPS |
| 양자화: FP16 | **부분 가능** | fp32 fallback 2곳이 필요하고, 정확도 손실은 없음. 속도 이득은 backbone fp16에서만 나옴 (repo 90.8ms가 최적) |
| 양자화: INT8 | **불가능 (현 환경)** | 툴체인 없음, 커스텀 연산(DAF)과 flash-attn에 INT8 경로 없음 |

- 3090 한 장에서 **가지치기 + 미세조정 학습과 FP16 추론은 가능**합니다. QAT는 배치를 줄이면 될 가능성이 높지만 실측이 필요하고, INT8 배포는 툴체인 설치와 커스텀 연산 처리가 먼저 해결돼야 합니다.
- 경량화 관점에서 이번에 찾은 가장 큰 이득은 가지치기도 양자화도 아닌 **`InstanceQueue.prepare_motion` 연산 교체**입니다. 재학습 없이 연산 결과가 동일하면서 지연 −10.6ms(−11%), peak −388MB(fp32) / −754MB(full_amp)입니다. 적용하려면 `instance_queue.py`를 수정해야 해서 이번에는 반영하지 않았고, 적용 전에 전체 검증셋 평가로 한 번 더 확인하는 것을 권합니다.

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
