# outputs_A — 실험 A 원시 출력 (prepare_motion matmul 교체 검증)

검증셋 전체(nuScenes v1.0-trainval val, 6,019 프레임)의 프레임별 모델 출력입니다. 실험 A(REPORT.md)의 샘플 단위 비교에 썼습니다. pkl 파일은 `.gitignore`로 git에서 제외되어 있어서 이 README만 커밋됩니다.

## 파일

| 파일 | 내용 | 원래 이름 | 생성 시각 (KST) | 크기 | sha256 |
|---|---|---|---|---:|---|
| `base_run1.pkl` | 원래 `prepare_motion` (브로드캐스트 식), 1회차 | `outputs/queue_eval/original/outputs.pkl` | 2026-10-22 11:30:49 | 2,385,776,685 B | `b70ea7b2fc262fa927a6461aa995933d666fb781a8d184c805dcd7d187c1148c` |
| `base_run2.pkl` | 원래 `prepare_motion`, 2회차 (실행 간 잡음 하한용) | `outputs/queue_eval/original_2/outputs.pkl` | 2026-10-22 13:07:23 | 2,385,833,263 B | `be2452197b61def46c5b72077ca28b939c48aadc9a39910f69cf2a20edbfa237` |
| `matmul.pkl` | `prepare_motion`의 브로드캐스트 합을 matmul로 교체 | `outputs/queue_eval/matmul/outputs.pkl` | 2026-10-22 12:18:46 | 2,385,704,062 B | `cb4210aeeea32a9b7d86b6914894766c6eeaf00161d0b810d72b72cd31ee7f10` |

생성 순서: base_run1 → matmul → base_run2. 2026-10-23에 이 폴더로 옮기고 이름을 바꿨습니다(내용은 그대로).

## 생성 환경

- 저장소 커밋: `c2aeb1945666ef5f0cb119a24c92a55934ddc4c7` (브랜치 `backbone_r18`, "feat: ResNet18 백본 student 설정 추가"). 모델·config·학습 코드는 이 커밋 그대로입니다.
- 측정 스크립트 `tools/feasibility/`는 당시 **커밋되지 않은 상태**였습니다. 지금의 `queue_matmul_eval.py`에는 이후 `--config`, `--no-save` 옵션이 추가됐지만, 기본값으로 실행하면 당시와 동작이 같습니다.
- config `projects/configs/stage2/distilldrive_stage2_distribution.py`, 체크포인트 `checkpoint/distilldrive_stage2_distribution.pth`
- 배포 모델: `distillation_head` 제외 (`with_distillation=False`)
- 정밀도: repo 모드 (`tools/test.py`와 같음. `wrap_fp16_model` → backbone·neck만 fp16)
- 시드 0. 단, 평가는 결정적이지 않습니다. DAF `atomicAdd`와 평가 중 `randn_like` 샘플링 때문이며, 자세한 내용은 REPORT.md "평가 비결정성" 절에 있습니다.
- 플래그: `cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=True`, `cudnn.benchmark=False`, `cudnn.deterministic=False`
- torch 1.13.0+cu116, RTX 3090

## 생성 명령 (당시 `tools/feasibility/run_queue_eval.sh`, 2026-10-28 삭제 — 커밋 `81241a2`에 있음)

```bash
cd /workspace/yujung/DistillDrive && source .venv/bin/activate && export PYTHONPATH=$(pwd)
python tools/feasibility/queue_matmul_eval.py --variant original --run original               # → base_run1.pkl
python tools/feasibility/queue_matmul_eval.py --variant matmul   --run matmul                 # → matmul.pkl
python tools/feasibility/queue_matmul_eval.py --variant original --run original_2 --no-metrics  # → base_run2.pkl
```

## 지표 (repo 모드, 검증셋 전체)

| 파일 | L2 (m) | obj_box_col | NDS | Det mAP | Map mAP | 추론 시간 (전체) |
|---|---:|---:|---:|---:|---:|---:|
| base_run1 | 0.7125 | 0.257% | 0.3810 | 0.2996 | 0.4207 | 10.6분 |
| matmul | 0.7121 | 0.255% | 0.3806 | 0.2984 | 0.4210 | 9.4분 |
| base_run2 | 평가 안 함 (출력만 저장) | | | | | 10.3분 |

match 행렬은 세 실행 모두 6,018회 호출 동안 `torch.bool`, 값 {0, 1}, 행·열당 최대 1개였습니다.

## 형식과 읽는 법

`mmcv.dump`로 저장한 리스트이고 길이는 6,019입니다. 원소는 `{"img_bbox": {...}}` 형태로 `boxes_3d`, `scores_3d`, `labels_3d`, `cls_scores`, `instance_ids`, `vectors`, `scores`, `labels`, `trajs_3d`, `trajs_score`, `anchor_queue`, `period`, `planning_score`, `planning`, `final_planning`, `ego_period`, `ego_anchor_queue`를 담습니다. `instance_ids`가 CUDA 텐서로 저장돼 있어서, 읽을 때 GPU가 보이는 환경이어야 합니다.

```python
import mmcv
runs = {n: mmcv.load(f"tools/feasibility/outputs_A/{n}.pkl") for n in ["base_run1", "base_run2", "matmul"]}
```

샘플 단위 비교 결과는 `outputs/queue_eval/compare_noise_floor.json`(base_run1 대 base_run2)과 `compare_orig_vs_matmul.json`(base_run1 대 matmul)에 있습니다. 둘 다 `compare_outputs.py`(2026-10-28 삭제, 커밋 `81241a2`에 있음)로 만들었습니다.
