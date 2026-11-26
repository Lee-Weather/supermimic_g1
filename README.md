# G1 SuperMimic 训练工作区

Unitree G1（29 DoF）的行走 tracking 训练：代码 + 数据集 + 云端启动脚本，一个仓库搞定。

## 目录

```
.
├── supermimic/            # 训练代码（依赖 mjlab）
├── mjlab_research/        # mjlab（supermimic 通过 ../mjlab_research 路径依赖）
├── datasets/
│   ├── walk_only/         # ★ 训练数据集：63 条 / 37.2 min（npz + SHA256SUMS）
│   ├── g1_ase_45/         # UMR/ACCAD 原始子集（转向/后退/起步/站立）
│   ├── tools/             # 转换、校验、接地、审计、渲染脚本
│   ├── manifest.json      # 机器可读清单（209 条 / 329.8 min / 0 problems）
│   └── README.md          # 数据格式契约与来源说明
└── cloud/
    ├── bootstrap.sh       # 一次性：uv + Python 3.13 + torch cu126 + mjlab + 冒烟
    ├── start_train.sh     # 训练入口（平台 startScript 用它）
    └── README.md          # 平台参数（goodsId/imageId/startScript）与风险
```

`datasets/source/`、`datasets/lafan1_g1/`、`datasets/walk_gapfill/` 是指向
`E:\G1\datasets\...` 的 junction，**不入库**；`datasets/*/packed/` 是派生物，容器内由
`pack_motions.py` 生成。细节见 [cloud/README.md](cloud/README.md)。

## 本地训练

```bash
# 依赖 uv（https://docs.astral.sh/uv/）：Python 3.13 由 uv 自动安装
uv venv --python 3.13 supermimic/.venv
source supermimic/.venv/bin/activate
uv pip install --extra-index-url https://pypi.nvidia.com -e mjlab_research -e supermimic

python supermimic/src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --num-envs 512 \
  --motion-file datasets/walk_only/npz \
  --runner.max-iterations 100000
```

Windows 下如需 GPU 版 torch，先单独装：
`uv pip install torch --index-url https://download.pytorch.org/whl/cu126`

## 云端训练

```bash
bash cloud/bootstrap.sh                       # 装环境 + 20 iter 冒烟（约 6-10 min）
# 平台的 startScript：
gm-run cloud/start_train.sh --num-envs 1024 --iters 120000
```

平台参数（算力 SKU、镜像、`codeUrl`/`mainCodeUri`/`startScript` 字段）与已知风险见
[cloud/README.md](cloud/README.md)。

## 数据

| 数据集 | clips | 时长 | 说明 |
|---|---|---|---|
| `walk_only` | 63 | 37.2 min | 直行 33.2 min + 转向 2.6 + 后退 0.75 + 起步 0.44 + 站立 0.25 |
| `g1_ase_45` | 45 | 4.9 min | 含 run/跳跃，非 walk-only 部分不参与行走训练 |

格式：`joint_pos / joint_vel / root_pos_w / root_quat_w (wxyz) / root_lin_vel_w /
root_ang_vel_w / joint_names`，50 Hz，关节顺序 = mjlab 的 G1 29-DoF 顺序。
来源：ACCAD（UMR 重定向）+ CMU/HDM05（UMR 重定向），详见 [datasets/README.md](datasets/README.md)。

## 上游仓库

本仓库 vendor 了两份上游代码用于云端一次性 clone：

- `supermimic/` ← https://github.com/HybridRobotics/supermimic.git
- `mjlab_research/` ← https://github.com/HybridRobotics/mjlab_research.git

原始 `.git` 目录分别改名为 `.git.upstream` 保留在本地（不入库）；要恢复上游追踪：
`mv supermimic/.git.upstream supermimic/.git`。