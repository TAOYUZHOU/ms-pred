# MassSpecGym (`msg`) 数据目录

本目录**不随仓库同步**大文件。克隆后请按下列步骤在本地生成或下载数据。

## 1. 需要什么

训练脚本默认从 `data/spec_datasets/msg/` 读取（`dataset-name: msg`），与仓库内 `configs/*/*_baseline_msg.yaml` 一致。典型布局：

```text
data/spec_datasets/msg/
├── labels.tsv                    # 训练入口默认 --dataset-labels（见下文）
├── labels_withev.tsv             # 由 prepare 脚本生成（含 collision energy 列）
├── spec_files.hdf5               # 谱图 HDF5（体积大）
├── splits/
│   └── split.tsv                 # train / val / test
└── subformulae/
    ├── no_subform.hdf5           # 子式 HDF5（体积大；configs 里 form-dir-name 常指向此文件）
    └── no_subform/               # 可选：从 HDF5 解压出的逐条 .json（FFN/GNN/3DMolMS 等）
```

## 2. 获取途径

### 2.1 谱图与划分（推荐：本地从 MassSpecGym 表生成）

1. 获取官方 **MassSpecGym** 制表文件（如 `MassSpecGym.tsv`，字段需含 `identifier`、`mzs`、`intensities`、`fold`、`smiles` 等）。发布与引用见论文 [MassSpecGym](https://arxiv.org/abs/2410.23326) 及项目说明。
2. 在仓库根目录执行：

   ```bash
   cd /path/to/ms-pred
   python scripts/prepare_msg_data.py \
     --msg-tsv /path/to/MassSpecGym.tsv \
     --out-dir data/spec_datasets/msg
   ```

   将生成：`spec_files.hdf5`、`splits/split.tsv`、`labels_withev.tsv`。

3. 训练脚本默认读取 **`labels.tsv`**。若只有 `labels_withev.tsv`，可复制或软链：

   ```bash
   cp data/spec_datasets/msg/labels_withev.tsv data/spec_datasets/msg/labels.tsv
   # 或: ln -sf labels_withev.tsv data/spec_datasets/msg/labels.tsv
   ```

### 2.2 子式 `no_subform.hdf5`

- 可与 **ICEBERG / MassSpecGym** 相关资源一并获取：仓库根目录 [README.md](../../../README.md)「Data」与「Experiments」中提到的 [Dropbox 共享文件夹](https://www.dropbox.com/scl/fo/d73o0o4u5ymr9ubtp3m7j/AL4r7e3p9ElV0ewBwDCScbM?rlkey=tr99zkzy208ol8aw0pfsdsf5v&st=2zg9n01y&dl=0)（含权重、检索集等；若网络受限见 `troubleshooting.md`）。
- 若你已有 `subformulae/no_subform.hdf5`，且模型需要**按 JSON 目录**读子式，可从 HDF5 解压：

  ```bash
  python scripts/extract_subform_to_dir.py \
    --hdf5 data/spec_datasets/msg/subformulae/no_subform.hdf5 \
    --labels data/spec_datasets/msg/labels.tsv \
    --out-dir data/spec_datasets/msg/subformulae/no_subform
  ```

### 2.3 预训练权重（不在本目录）

权重、大 zip 等请放在 **`pretrained_weights/`** 等单独目录，并按根目录 README 说明获取；勿提交到 Git（见 `.gitignore`）。

## 3. 快速自检

```bash
test -f data/spec_datasets/msg/labels.tsv        && echo OK labels.tsv
test -f data/spec_datasets/msg/spec_files.hdf5   && echo OK spec_files.hdf5
test -f data/spec_datasets/msg/splits/split.tsv  && echo OK split.tsv
test -f data/spec_datasets/msg/subformulae/no_subform.hdf5 && echo OK no_subform.hdf5
```

## 4. 许可与引用

使用 MassSpecGym 数据请遵守其官方许可；发表工作请引用 MassSpecGym 与 **ms-pred** 原仓库论文（见根目录 README「Citation」）。
