# 인수인계: DistillDrive R50 경량화 (가지치기·양자화·회복 학습)

R50 stage2 distribution 모델의 경량화가 RTX 3090 1장에서 가능한지 재학습 없이 점검한 결과와, 이어서 진행하실 때 필요한 내용을 정리했습니다. 자세한 측정값과 근거는 같은 폴더의 [REPORT.md](REPORT.md)에 있습니다.

- 표시: **(확인)** = 실제로 측정하거나 코드로 확인한 내용, **(추정)** = 측정값을 바탕으로 한 판단
- 작성: 2026-10-28, 브랜치 `feasibility/r50-stage2`

---

## 1. 한 줄 요약

3090 1장에서 **학습(회복 학습 포함)과 평가 모두 메모리 여유 있게 돌아갑니다 (확인)**. 바로 쓸 수 있는 무손실 개선이 하나 있고(`prepare_motion` matmul 교체), decoder 층 수를 줄이는 가지치기는 **재학습 없이는 성능이 무너져서 회복 학습이 필요합니다 (확인)**. 회복 학습 설정은 제안까지만 해 두었고 실행하지 않았습니다.

---

## 2. 어디에 무엇이 있나

| 위치 | 내용 |
|---|---|
| GitHub `yuudengg/DistillDrive_3090`, 브랜치 `feasibility/r50-stage2` | 측정 도구, 결과(JSON·평가 로그), REPORT.md, 이 문서 |
| `tools/feasibility/REPORT.md` | 전체 실험 결과와 근거 (실험 1~3, A, B, 평가 비결정성, head 분해, 층 수 시험, 전체 결론) |
| `tools/feasibility/outputs/` | 결과 파일 |
| `tools/feasibility/configs/` | 층 수를 줄인 설정 4개 (기존 config 상속, `operation_order`만 자름) |
| 서버 `checkpoint/distilldrive_stage2_distribution.pth` | 기준 R50 stage2 학생 체크포인트 (GitHub에 없음) |
| 서버 `checkpoint/distilldrive_stage0_distribution.pth` | teacher (stage2 증류용, `dlp_checkpoint`) |
| 서버 `tools/feasibility/outputs_A/*.pkl` | 실험 A 원시 출력 3개, 약 7GB (`.gitignore`, 생성 정보는 같은 폴더 README.md) |

기존 config·체크포인트·학습 코드는 한 줄도 수정하지 않았습니다. 바꿔야 하는 부분(teacher 제외, 정밀도, 층 수, 연산 교체)은 전부 측정 도구 안에서 메모리상으로만 적용했습니다.

---

## 3. 환경 (서버) 주의사항

- **venv**: `/workspace/yujung/DistillDrive/.venv` (Python 3.8, torch 1.13.0+cu116, mmcv-full 1.7.1, flash-attn 2.3.2). venv 활성화·`PYTHONPATH` 설정·실행을 **같은 셸 명령 안에서** 해야 합니다.
- **GPU 1장 실행**: 학습은 `tools/dist_train.sh <config> 1 ...` (분산 런처로 1프로세스). `scripts/train.sh`는 `--deterministic` 외의 **추가 인자를 넘기지 않으므로**, `--no-validate` 같은 옵션이 필요하면 `tools/dist_train.sh`를 직접 부르세요.
- **할당기 설정**: 실제 stage2 학습은 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`로 돌렸습니다.
- **RAM 30GB**: 학습 데이터로더(`workers_per_gpu=4`, 늘리면 RAM 부족으로 SIGKILL 이력)와 검증셋 평가(프로세스 약 14GB)를 **동시에 돌리지 마세요**. 순서대로 실행을 권합니다.
- **커스텀 CUDA 연산 DAF** 재빌드가 필요하면 별도 conda 툴체인을 씁니다. 시스템 nvcc 12.8 / gcc 11.4는 torch의 cu116 빌드와 맞지 않습니다. `CXX`는 설정하지 말고 `CC`만 바꿔야 합니다.
  ```bash
  export CUDA_HOME=/home/jinholee/miniconda3/envs/cuda116-toolchain
  export PATH=$CUDA_HOME/bin:$PATH
  export CC=$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc
  cd projects/mmdet3d_plugin/ops && python setup.py develop
  ```

---

## 4. 기준값

**정확도** (repo 모드 = `tools/test.py`와 같은 방식, 검증셋 전체)

| | L2 (m) | obj_box_col | NDS | Det mAP | Map mAP |
|---|---:|---:|---:|---:|---:|
| 기준 (`test.py`) | 0.7130 | 0.259% | 0.3803 | 0.2990 | 0.4207 |
| **평가 흔들림 범위 폭** (같은 코드 4회) | 0.0012 | 0.005%p | 0.0007 | 0.0006 | 0.0002 |

**평가는 결정적이지 않습니다 (확인)**. 같은 체크포인트도 실행마다 지표가 위 폭만큼 흔들립니다. 원인은 DAF 커널의 `atomicAdd`와 평가 중 `torch.randn_like` 샘플링(motion_plan_head)이고, 시드로 통제되지 않습니다. **이 폭보다 작은 차이는 변경 효과로 보지 마세요.** `test.py --deterministic`도 `--seed`를 줘야만 적용됩니다.

**속도·메모리** (이 측정 도구 기준, 배치 1 추론 / 배치 4 학습)

| | 값 |
|---|---|
| 추론 지연 fp32 / repo 모드 | 99.0 ms / 90.8 ms (예전에 받은 110.6ms는 다른 측정 방법 값이라 비교에 쓰지 않음) |
| 추론 peak allocated (fp32) | 1,245 MB |
| 학습 peak allocated / 프로세스 전체 (측정, 6층) | 10,558 MB / 16,554 MB |
| 학습 peak (실제 stage2 학습 로그) | 11,245 MB — 측정보다 약 6% 높음, 70k iter 내내 거의 일정 (확인) |
| 학습 iter 시간 (실제 로그) | 0.965 s/iter |

**정밀도**: stage2 학습과 `test.py` 평가는 둘 다 혼합 정밀도입니다. config의 `fp16=dict(loss_scale=32.0)` 때문에 backbone·neck·depth만 fp16(autocast), head·DAF·loss는 fp32, 가중치는 fp32 (확인).

---

## 5. 꼭 알아야 할 함정

1. **`resume_from` 자동 설정**: stage2 config는 `work_dirs/distilldrive_stage2_distribution/latest.pth`가 있으면 자동으로 이어서 학습합니다. 이 파일이 **실제로 있습니다**(iter 70,320). 새 config에서 `resume_from = None`을 꼭 덮어쓰세요. 안 그러면 6층 학습 상태에서 이어서 시작합니다.
2. **층을 자르면 하류가 깨짐**: det/map head의 마지막 층 출력이 instance bank 캐시(다음 프레임 temporal attention)와 motion_plan_head로 들어갑니다. 층을 줄이면 이 입력이 학습 때와 달라져서 성능이 크게 떨어집니다(아래 6절). 장면 전환 프레임에서도 이전 장면 캐시가 temporal attention 키로 쓰입니다.
3. **map decoder는 `output_idx`를 무시**: 항상 `[-1]` 층을 씁니다 (`projects/mmdet3d_plugin/models/map/decoder.py` 27-29행). 중간 층 예측을 평가하려면 `common.read_eval_layer`처럼 층 목록을 잘라서 넘겨야 합니다.
4. **FP16을 head까지 쓰려면**: `DistillDrive.forward`의 `@force_fp32`가 바깥 autocast를 꺼서, head 단에서 autocast를 다시 켜야 합니다. 그러면 장면 전환 프레임(Δt 약 −43만 초)에서 `anchor_projection`과 `anchor_encoder`가 fp16 범위를 넘어 NaN이 나고, 캐시를 타고 이후 모든 프레임으로 퍼집니다. 이 두 곳만 fp32로 두면 해결됩니다(`common.apply_precision(..., "full_amp")`). 또 `post_process`의 `@force_fp32(apply_to=("model_outs"))`는 인자 이름이 안 맞아 실제로 캐스팅하지 않습니다.
5. **`InstanceQueue.prepare_motion` 병목**: 추적 매칭을 900×900×256 브로드캐스트로 계산해서 텐서 829MB(배치 4면 3.2GB)를 매번 만듭니다. matmul로 바꾸면 연산·기울기 모두 결과가 동일하고, 추론 −10.6ms, 학습 peak −2.1GB, iter −5.4%입니다 (확인). **소스에는 반영하지 않았습니다.** 측정 도구에서는 `common.set_queue_variant("matmul")`로 켭니다. 실제 학습에 쓰려면 소스 수정이나 별도 래퍼가 필요합니다. 참고로 실제 학습에서 이 연산으로는 기울기가 흐르지 않습니다(큐 텐서가 `detach`됨).
6. **INT8 불가 (현 환경)**: TensorRT·pytorch-quantization·ONNX가 없고, torch 내장 양자화는 CPU 전용입니다. DAF는 fp32 전용 커널, flash-attn은 fp16 전용이라 INT8 경로가 없습니다.

---

## 6. 지금까지의 결론 (자세한 표는 REPORT.md)

**양자화 (확인)**
- FP16: fallback 2곳만 두면 정확도 손실 없음. 다만 속도 이득은 이미 기본 평가가 쓰는 backbone fp16에서 다 나오고, head까지 fp16이면 오히려 1.3ms 느림 → repo 모드가 최적.
- INT8: 위 5-6 때문에 막힘.

**가지치기 대상 분석 (측정 확인, 가지치기 효과는 추정)**

| 연산 | det_head 학습 메모리 / 추론 시간 비중 | map_head 비중 | Torch-Pruning 구조 판단 (코드 분석, 추정) |
|---|---|---|---|
| deformable (커스텀 CUDA DAF) | 37% / 48% | 94% / 86% | 불가에 가까움 (FPN 특징 채널 256·그룹 8과 결합) |
| self + temporal attention | 23% / 17% | 1.5% / 5% | 어려움 (flash-attn, 원시 `in_proj_weight` 사용 → 사용자 정의 pruner 필요) |
| FFN (512→**1024**→256) | 12% / 6% | 2% / 2% | 가능 (표준 MLP, 은닉 차원만) |
| refine, anchor encoder | 27% / 25% | 2.4% / 4% | 내부 차원만 가능 |
| 잔차 스트림 256 | – | – | 불가 (모든 head·DAF·FPN 공통) |

- FFN 은닉 차원을 40% 잘라도 학습 메모리 −1.0%, 추론 지연 −0.6% 이하 (추정).
- Torch-Pruning은 설치하지 않았고 의존성 추적도 돌리지 않았습니다. 위 판단은 코드 구조만 보고 한 것입니다.

**decoder 층 수 축소 (재학습 없음, 확인)**

| 설정 | L2 | NDS | Det mAP | Map mAP | 추론 지연 | 학습 peak |
|---|---|---|---|---|---|---|
| 6층 (기준, matmul) | 0.7121 | 0.3806 | 0.2984 | 0.4210 | 79.9 ms | 8,431 MB |
| 4층 자르기 | 1.2835 | 0.2497 | 0.1077 | 0.2824 | 65.8 ms | 6,926 MB |
| 6층 실행, 평가용 det/map만 4층 출력 (진단) | 0.7131 | 0.3655 | 0.2939 | 0.4171 | – | – |

- 진단 결과, 4층 예측 자체의 손실은 Det/Map mAP −0.004~0.005 수준입니다. 자르기 하락의 대부분은 **하류 입력 분포 차이**에서 옵니다. 그래서 회복 학습으로 상당 부분 되찾을 여지가 있습니다 (추정).
- 학습 로그상 층 3~4 이후 손실이 거의 줄지 않습니다 (확인).
- 한쪽만 자르면: det만 4층 → L2 +0.035, map만 4층 → L2 +0.354. planning은 map 절단에 훨씬 민감합니다 (확인).

---

## 7. 다음 단계 제안: 4층 회복 학습 (실행 안 함)

**config 초안** — `tools/feasibility/configs/recover_stage2_distribution_layers4.py`로 새로 만들기를 권합니다.

```python
_base_ = ["./stage2_distribution_layers4.py"]          # det/map 4층, 나머지는 stage2 distribution 그대로
load_from = "checkpoint/distilldrive_stage2_distribution.pth"   # 6층 학생 가중치, 잘라낸 층 키는 무시됨 (strict=False)
resume_from = None                                      # 필수 (5절-1)
num_epochs = 3
checkpoint_config = dict(interval=7032)                 # 에폭마다 저장 (28130 // 4)
runner = dict(type="IterBasedRunner", max_iters=7032 * num_epochs)
optimizer = dict(lr=1e-4)
lr_config = dict(policy="CosineAnnealing", warmup="linear", warmup_iters=500,
                 warmup_ratio=1 / 3, min_lr_ratio=1e-3)
```

- 그대로 두는 것: 배치 4 × 누적 12(유효 48), AdamW, backbone `lr_mult=0.1`, grad clip 25, fp16(loss_scale 32), `with_cp=True`, teacher와 증류 손실 전부.
- **에폭 수**: 3(권장) 또는 2. cosine은 정한 길이 끝에서 학습률이 0 근처가 되므로, 2로 했다가 늘리기는 깔끔하지 않습니다. 3이면 1·2에폭 중간 평가로 방향을 볼 수 있습니다.
- **학습률**: 1e-4(권장) 또는 원래 값 3e-4. 이미 수렴한 가중치라 하류 적응만 필요해서 낮게 두는 쪽을 권합니다. 3e-4는 1e-4로 부족할 때 시도.
- **`prepare_motion`**: 원래 식(소스 수정 없음) 또는 matmul(메모리·속도 이득, 소스 수정 필요). 결과는 같습니다.
- 위 세 가지는 아직 정하지 않았습니다.

**실행과 평가**
```bash
cd /workspace/yujung/DistillDrive && source .venv/bin/activate && export PYTHONPATH=$(pwd) \
  && export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  && bash tools/dist_train.sh tools/feasibility/configs/recover_stage2_distribution_layers4.py 1 --deterministic --no-validate
```
- 학습 중 평가 훅은 이 저장소에서 써 본 적이 없어서, `--no-validate`로 끄고 에폭 체크포인트마다 `queue_matmul_eval.py`로 따로 평가하길 권합니다. 그래야 기준·자르기·진단 결과와 같은 조건으로 비교됩니다.
- **`queue_matmul_eval.py`에 `--ckpt` 옵션 추가가 필요합니다.** 지금은 기준 체크포인트로 고정이고, `common.build_model(cfg, ckpt=...)`은 이미 받으므로 인자만 넘기면 됩니다.

**예상 (추정)**

| | 학습 | 에폭별 평가 (각 약 48분, 측정) | 합계 |
|---|---|---|---|
| 2에폭 | 약 3.0시간 | 약 1.6시간 | 약 4.6시간 |
| 3에폭 | 약 4.5시간 | 약 2.4시간 | 약 6.9시간 |

- 학습 메모리 (실제 학습 보정 +6% 반영): 원래 식이면 peak 약 9.7GB / 프로세스 약 16.5GB, matmul이면 약 7.6GB / 약 13.5GB. 24GB 안에 들어갑니다.

**판단 기준**
- 잘 가는 신호: 에폭마다 L2가 0.7121 쪽으로 내려가고, Det/Map mAP가 진단 값(0.294 / 0.417) 쪽으로 오름.
- 학습률이 너무 높은 신호: 1에폭에서 Det/Map mAP가 자르기 값(0.108 / 0.282)보다 더 떨어지거나 손실이 튐.
- 학습률이 너무 낮은 신호: 1에폭에서 L2가 1.28 근처에서 거의 안 움직임.
- 비교할 때는 4절의 평가 흔들림 폭을 함께 보세요.

---

## 8. 측정 도구

| 파일 | 용도 |
|---|---|
| `common.py` | 설정 로드, 배포 모델 구성(teacher 제외, 잘라낸 층 키 무시), 정밀도 모드(fp32 / repo / full_amp_raw / full_amp), `prepare_motion` 원래·matmul 전환, 평가용 층 지정 |
| `train_memory.py` | 학습 메모리·iter 시간. `--config --with-cp --queue --hooks [heads] --tag` |
| `infer_bench.py` | 추론 지연·메모리, `--eval`이면 검증셋 전체 평가. `--mode --config --queue --tag` |
| `module_memory.py` | 모듈별 추론 활성값 |
| `queue_matmul_eval.py` | 검증셋 전체 평가 (repo 모드, 시드 고정). `--variant --config --run --no-save --eval-layer` |
| `head_breakdown.py` | det/map head 층·연산별 추론 시간과 학습 메모리 요약 |
| `layer_losses.py` | 학습 로그에서 층별 손실 표 (회복 학습 로그에도 `--log`로 사용 가능) |
| `probe_layer_exit.py` | 잘린 모델 vs 6층 모델의 해당 층 출력 비교 (회복 학습 뒤 하류 입력 차이 확인용) |
| `configs/stage2_distribution_{layers4,layers5,det4_map6,det6_map4}.py` | 층 수 축소 설정 |

사용 예시는 REPORT.md의 "재현" 절에 있습니다. 결론이 나온 일회성 진단과 일괄 실행 스크립트는 정리했고, 필요하면 커밋 `81241a2`에서 꺼낼 수 있습니다.
