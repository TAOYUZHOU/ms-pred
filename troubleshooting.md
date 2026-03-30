# ms-pred (ICEBERG) 踩坑记录

> 格式：一句话描述问题 → 一句话描述解决方案

---

### 1. environment.yml 要求 CUDA 11.1 + Python 3.8，与现代 GPU 环境不兼容

**问题：** 官方 `environment.yml` 锁定 `pytorch=1.9.1+cuda11.1`、`dgl-cuda11.1=0.8.2`、`python=3.8`，但系统 CUDA 为 12.8，conda 根本找不到这些旧版包。

**方案：** 放弃 `environment.yml`，手动创建 Python 3.10 环境，用 pip 安装兼容 CUDA 12.1 的版本：
```bash
conda create -p <env_path> python=3.10 -y
<env_path>/bin/pip install torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
<env_path>/bin/pip install pytorch-lightning==1.9.5 dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
<env_path>/bin/pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

---

### 2. setup.py Cython 编译失败：algos2.pyx 中 `long` 未定义

**问题：** `src/ms_pred/massformer_pred/massformer_code/algos2.pyx` 中使用了 `long` 类型，Python 3.10+ 中 `long` 已移除，Cython 编译报 `undeclared name not builtin: long`。

**方案：** 在 `setup.py` 中将 `cythonize(...)` 包裹在 `try/except` 中，编译失败时跳过 MassFormer 扩展。不影响 ICEBERG/SCARF/MARASON 等核心模型。如确需 MassFormer，可将 `algos2.pyx` 中 `long` 改为 `int`。

---

### 3. conda run -p 调用了 base 环境的 pip/python

**问题：** `conda run -p <prefix> pip install ...` 实际使用 base 环境的 `/root/miniconda3/bin/pip`，包装到了 base，conda env 里啥都没有。

**方案：** 直接用绝对路径调用：`<env_path>/bin/pip install ...` 和 `<env_path>/bin/python ...`。

---

### 4. import ms_pred.dag_pred 失败：No module named 'torch_scatter'

**问题：** `nn_utils.py` 依赖 `torch_scatter`，但 pip 默认不会从 PyG 源安装。

**方案：** 从 PyG wheel 源安装：
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

---

### 5. import ms_pred.nn_utils 失败：No module named 'ray'

**问题：** `tune_utils.py` 顶层 `from ray.tune.integration.pytorch_lightning import TuneCallback`，即使不做超参搜索也会触发。

**方案：** 安装 ray：`pip install "ray[tune]"`。

---

### 6. NumPy 2.x 与 torch 2.2.0 ABI 不兼容

**问题：** 默认安装的 `numpy>=2.0` 与 `torch 2.2.0` 编译时用的 NumPy 1.x ABI 不兼容，报 `_ARRAY_API not found`。

**方案：** 降级 numpy：`pip install "numpy<2"`（会安装 1.26.x）。

---

### 7. spec_files.hdf5 的字符串存储格式：read_str 要求 bytes 数组

**问题：** `common.HDF5Dataset.read_str()` 使用 `grp[name][0]` 读取并要求 `type(str_obj) is bytes`。如果用 `f.create_dataset(name, data="string")` 存的是标量 numpy.bytes_，会报 `Illegal slicing argument for scalar dataspace` 或 `Wrong type`。

**方案：** 写入时必须用 vlen bytes 数组格式：
```python
dt = h5py.special_dtype(vlen=bytes)
f.create_dataset(name, data=np.array([content.encode("utf-8")], dtype=object), dtype=dt)
```

---

### 8. MAGMa 片段化产出 0 棵树

**问题：** 如果 spec_files.hdf5 中的峰位置是完全随机的（与分子的实际片段质量不匹配），MAGMa 无法将任何峰匹配到片段，产出空树。

**方案：** 在生成合成谱图时，使用 `FragmentEngine` 先计算分子的真实片段质量，再以这些质量（加上加合物质量偏移）作为峰的 m/z 值。参见 `smoke_test.py` 中的 `create_spec_files_hdf5()` 实现。

---

### 9. NIST20 数据集不完整 / MassSpecGym 数据缺失

**问题：** 仓库中 `data/spec_datasets/nist20/` 只有 `splits/` 目录，缺少 `labels.tsv`、`spec_files.hdf5`、`magma_outputs/`。`msg/` 目录只有 `labels.tsv`，缺少其余文件。

**方案：** NIST20 是商业数据集，需购买后按 README 指引导出 SDF 并处理。MassSpecGym 权重和数据可从 [Dropbox](https://www.dropbox.com/scl/fo/d73o0o4u5ymr9ubtp3m7j/AL4r7e3p9ElV0ewBwDCScbM?rlkey=tr99zkzy208ol8aw0pfsdsf5v&st=2zg9n01y&dl=0) 下载。Smoke test 可用合成数据验证 pipeline，参见 `smoke_test.py`。

---

### 10. pytorch_lightning 1.6 vs 1.9 Logger API 差异

**问题：** `misc_utils.py` 同时兼容 PL 1.6 和 1.9 的 Logger 导入路径，但如果安装的 PL 版本不在这两个之内可能仍会出问题。

**方案：** 锁定 `pytorch-lightning==1.9.5`，代码中已有 `try/except` 处理两种导入路径。

---

### 11. Dropbox 预训练权重无法从国内网络下载

**问题：** README 提供的 MassSpecGym 预训练权重在 Dropbox 共享文件夹中，国内服务器（如 AutoDL）无法直接访问 Dropbox（连接超时、HTTP 000）。

**方案：** 可通过以下途径获取权重：① 有海外代理时使用代理下载；② 有 NIST'20 许可证时邮件联系作者获取 NIST20 版权重；③ 用 smoke test 训练小模型跑通 pipeline 验证流程，然后在有网络条件的环境下载正式权重。

---

### 12. MAGMa 树缺少 `raw_spec` 字段，IntenGNN 训练时 KeyError

**问题：** `magma_augmentation()` 产出的树只包含 `root_canonical_smiles`、`frags`、`collision_energy`、`adduct`，没有 `raw_spec`。IntenGNN 的 `_convert_to_dgl()` 在 `include_targets=True` 时读取 `tree["raw_spec"]`，导致 `KeyError: 'raw_spec'`。

**方案：** 需要用 `data_scripts/dag/add_dag_intens.py --add-raw` 将原始谱图的 m/z + intensity 写入树的 `raw_spec` 字段。对 smoke test 合成数据，直接在生成树时从 `spec_files.hdf5` 解析峰数据并添加 `tree["raw_spec"] = list(zip(mzs, intens))`。

---

### 13. IntenGNN 训练参数必须匹配官方配置：`binned_targs=True`, `include_unshifted_mz=True`

**问题：** 使用 `binned_targs=False` + `include_unshifted_mz=False` 训练 IntenGNN 时，`_common_step()` 中 `torch.stack((batch["masses"], pred_inten), dim=-1)` 报维度不匹配——`masses` 为 `[B, N, 2, 13]`（含 shifted+unshifted），`pred_inten` 为 `[B, N, 13]`。

**方案：** 按官方配置 `configs/iceberg/dag_inten_train_nist20.yaml` 设置 `binned_targs=True` 和 `include_unshifted_mz=True`。这是 ICEBERG 设计的预期训练路径。

---

### 14. MAGMa 树中 collision_energy 为 nan，instrument 字段缺失

**问题：** `magma_augmentation()` 生成的树名为 `{spec_id}_collision nan.json`，因为 `collision_energies` 字段格式为字符串列表（如 `"['30']"`），MAGMa 解析时未正确提取数值。同时树中缺少 `instrument` 字段，导致使用 `embed_instrument=True` 时报错。

**方案：** 在使用树之前，从 `labels.tsv` 读取正确的 collision_energy 和 instrument，更新树的对应字段并重命名（如 `MassSpecGymID0000000_collision 30.0.json`）。

---

### 15. predict_mol() 中 `add_pe_embed` 无条件调用导致维度不匹配（关键 Bug）

**问题：** `gen_model.py` 的 `predict_mol()` 方法在第 542 行和第 631 行无条件调用 `self.tree_processor.add_pe_embed()`，即使 `pe_embed_k=0`。但训练时 `_process_tree()` 仅在 `pe_embed_k > 0` 时才调用。`random_walk_pe(g, k=0)` 会返回 1 维特征并拼接到节点特征中，导致预测时输入维度比训练多 1（例如 118 vs 117），报 `RuntimeError: mat1 and mat2 shapes cannot be multiplied`。

**方案：** 在 `gen_model.py` 的 `predict_mol()` 中添加条件判断：
```python
# 第 542 行附近
if self.pe_embed_k > 0:
    self.tree_processor.add_pe_embed(root_repr)

# 第 631 行附近
if self.pe_embed_k > 0:
    self.tree_processor.add_pe_embed(frag_batch)
```
此修改使预测行为与训练保持一致。

---

### 16. splits 文件列名格式要求：需要 `spec` 和 `fold_0` 列

**问题：** `common.get_splits()` 读取 split TSV 文件时，要求列名为 `spec`（样本ID）和 `fold_0`/`fold_1`（划分标签，值为 `train`/`val`/`test`）。如果 split 文件使用 `name` 和 `split` 作为列名，`get_splits()` 找不到正确的列导致划分失败。

**方案：** 确保 split 文件列名为 `spec` + `fold_0`（或 `fold_N`），值为 `train`/`val`/`test`。生成 smoke test splits 时按此格式创建。

---

*持续更新：遇到新问题时追加到本文档末尾。*
