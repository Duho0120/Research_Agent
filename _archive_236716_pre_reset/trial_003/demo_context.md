# 236716 / trial_003 Demo Context

- status: ready
- platform: dacon
- metric: R-Hit@1cm
- objective: maximize
- project_root: C:\Users\ASUS\Desktop\Research_Agent\demo_workspaces\236716

## Model Policy

- experiment_planning: openai / gpt-5 (high_cost)
- workspace_code_writing: openai / gpt-5 (high_cost)
- low_cost: openai / gpt-5.6-luna

## Commands

- test: `{python} test_step.py`
- train: `{python} train_step.py`
- predict: `{python} predict_step.py`

## Artifacts

- metrics: outputs/metrics.json
- submission: outputs/submission.csv

## Artifact Policy

- trained model default save: False
- primary memory: metrics, submission, code snapshot, pipeline summary

## RAG Context Pack

- task: experiment_planning
- documents: 0
- context_pack: `None`
- manifest: `None`

## Competition Documents

### overview.md

# 236716 - 모기 비행 궤적 예측 AI 경진대회

LiDAR 센서로 관측된 모기의 과거 좌표를 바탕으로 80ms 후의 3D 위치를 예측하는 시계열 회귀 문제.

- 입력: 40ms 간격으로 관측된 11개 시점의 3D 좌표 (x, y, z), 시간 범위 -400ms ~ 0ms
- 출력: +80ms 시점의 3D 좌표 (x, y, z)
- 좌표계: 방 기준 절대 좌표가 아니라 LiDAR 센서 기준 sensor-local 3D 좌표계. 단위는 미터(m). 축 방향은 x: forward, y: left, z: up.
- 평가지표: R-Hit@1cm. 예측 좌표와 실제 좌표의 3D 유클리드 거리 d를 계산해서 d <= 0.01m이면 적중(1), 아니면 실패(0). 점수는 전체 샘플의 평균 R-Hit(적중률).

### data_notes.md

# Data Notes

**IMPORTANT: there is no flat data/train.csv or data/test.csv file. Do not assume one exists.**

```
data/
├── train/                  # 10000 files, one per training sample
│   ├── TRAIN_00001.csv     # columns: timestep_ms, x, y, z (11 rows, -400ms..0ms in 40ms steps)
│   ├── TRAIN_00002.csv
│   ├── ...
│   └── TRAIN_10000.csv
├── test/                   # 10000 files, same structure as train/, one per test sample
│   ├── TEST_00001.csv
│   ├── ...
│   └── TEST_10000.csv
├── train_labels.csv        # columns: id, x, y, z -- id values are "TRAIN_00001" etc.,
│                            # matching the train/*.csv filename stem. x,y,z here is the
│                            # TARGET: the mosquito's actual position 80ms after the last
│                            # observed timestep in that sample's file.
└── sample_submission.csv   # columns: id, x, y, z -- id values are "TEST_00001" etc.
```

**How to build the training set:** for each `train/TRAIN_XXXXX.csv`, load its 11-row time
series (features), then look up the target `x,y,z` from `train_labels.csv` where
`id == "TRAIN_XXXXX"`. Predictions for `test/TEST_XXXXX.csv` go into `outputs/submission.csv`
keyed by `id == "TEST_XXXXX"`, matching `sample_submission.csv`'s row order/ids.

No leakage risk between train/test (disjoint sample folders). No column notes beyond the
above -- x,y,z are already numeric sensor-local coordinates in meters.

### metric.md

# Metric

- name: R-Hit@1cm
- objective: maximize


## Recent Trials

- trial_001: 1.0
- trial_002: 0.591
