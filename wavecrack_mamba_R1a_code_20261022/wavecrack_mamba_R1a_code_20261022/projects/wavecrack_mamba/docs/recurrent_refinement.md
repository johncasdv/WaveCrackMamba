# R1a 共享循环细化：实现与运行

2026-09-10。代码已实现并进行单元、集成和真实 DF 权重验证；未执行正式训练或精度评价。论文方案中的循环效果、机制和 HardCrack 收益仍待实验。

## 接入与兼容

`WCMInversePyramidHead` 默认 `refiner=None`，保留旧模型结构和权重键名。启用时在 `refine` 之后、最终四倍上采样之前接入 `WCMRecurrentRefiner`，通过原 `conv_seg` 形成粗掩膜，通过原 `cls_seg` 读出最终输出。

- 状态与输出为 N×128×H/4×W/4；Transformer 为 4 heads、8×8 特征窗口、FFN 128→256→128。
- T1 执行一次更新，T3 执行三轮共享更新，默认移位 0/4/0。
- K/V 只在单次 forward 内计算一次；不跨批次或优化步骤缓存学习中的投影。
- 更新投影与 FFN 末层零初始化，初始输出与旧基座一致。`steps=0` 跳过细化并返回原 F。
- `share_weights=False` 支持同构独立更新；`use_mask_feedback=False` 保留模块形状并将掩膜嵌入输入置零。对应归因实验按后续决议开展，不是首轮启动门槛。
- C3 使用共享 7×7 深度卷积，内部宽度160，状态和FFN仍为128。其固定特征经过一次 value 投影后与当前 query 相加，进入深度卷积、GELU及输出投影。
- 任意正尺寸的特征网格单独进行窗口填充、key屏蔽、移位边界隔离与裁剪；图像上下/左右边缘不会因循环 roll 产生虚假邻接。

新增可训练参数：T1/T3 为 **133,764**，C3 为 **136,704**。按512×512输入计主要算子MAC，T3约6.180G、C3约5.962G，后者低约3.5%；此估计不含归一化、激活等，不能当作实测等时延。

## 缓存特征与诊断接口

```python
features = model.decode_head.fuse_features(model.extract_feat(images))
logits = model.decode_head.predict_from_fused(features)
logits, coarse_logits = model.decode_head.predict_from_fused(
    features, steps=3, return_intermediates=True)
```

`coarse_logits` 包含初值及每轮更新后的1/4分辨率logits，仅作诊断，没有额外中间监督。不得把同一K=3模型的中间输出当作单独充分训练的K=1模型。

冻结基座时，可在 `torch.no_grad()` 内提取固定输入视图的 F；`predict_from_fused()` 必须位于可求导上下文，以保留状态经过旧分类器传回新模块的梯度。完整F缓存文件、数据集及其生成任务不在本次实现中。

## 冻结与 checkpoint

`WCMSegmentor(freeze_base_for_refiner=True)` 在优化器创建前关闭全部旧参数梯度。每次 `train()` 仍将旧模块置eval，仅细化器进入train；原分类器参数冻结，但对状态可微。

`WCMFrozenRefinerHook` 在模型初始化后加载旧基座；在复制参数前核对全部旧键名和形状，只允许缺失新 `decode_head.refiner.*`。旧checkpoint不允许混入部分refiner权重。

默认采用 `weights_only=True` 读取。旧MMEngine文件可能包含HistoryBuffer/NumPy训练元数据，只有调用方给出可信SHA-256且与读取的同一字节内容匹配，才允许兼容加载。不要对来源不明的文件绕过这项核验。

完整refiner checkpoint由标准 `tools/test.py CONFIG CHECKPOINT` 加载，不要求原DF文件仍存在。resume要求完整键名/形状核验且真正完成加载；空目录自动恢复明确报错，避免冻结随机基座训练。新训练禁止以普通 `load_from` 绕过旧基座的严格准入。

## 首轮配置与入口

配置位于 `configs/recurrent/`：`wcm_fz_t1_crack_v5.py`、`wcm_fz_t3_crack_v5.py`、`wcm_fz_c3_crack_v5.py`。

共同配方：旧域crack_v5、冻结DF、4,000更新步、batch8、AdamW lr=1e-4、200步warm-up、余弦衰减、梯度范数裁剪1，仅最终区域损失。checkpoint主报告为final，未启用best选择。

训练使用固定中心裁块作为单一可复现视图；小图交由已有预处理器填充。没有缩放整幅调查图，没有随机翻转、旋转或颜色增强。验证保持既有原图视图；pilot的test命令也使用验证划分，避免在候选选择时读取测试集。

`scripts/run_refiner_pilot.ps1` 参数：

- `-Arm t1|t3|c3`：要运行的候选。
- `-BaseCheckpoint`：可信旧DF checkpoint本地路径。
- `-BaseSha256`：可选已知哈希；给出时不一致即拒绝。省略时固定本次选定文件的哈希并传入加载器。
- `-DataRoot`：真实crack_v5数据根。
- `-TrainSplit`：相对于DataRoot的已核验训练子集列表，默认 `_diag/train_diag3k.txt`。
- `-ValSplit`：相对DataRoot的验证列表，默认 `splits/new_val_small.txt`。
- `-CudaVisibleDevices`：GPU数字列表或-1；通过WSLENV导出到Python环境。
- `-DryRun`：只检查路径和哈希并显示配置，不启动训练。

实际训练仍必须通过 **schtasks → run_refiner_pilot.ps1 → run_train.ps1 -Wsl**。包装器仅转交环境和参数，没有绕过已有调度纪律；本次未建立训练任务。调用前仍须核验训练诊断列表属于训练集且与验证不交叉，并登记实际图像、mask和划分版本。数据人工复核已完成，不重新设立人工复核门槛。

直接使用配置时可通过 `WCM_REFINER_DATA_ROOT`、`WCM_REFINER_BASE_CHECKPOINT`、`WCM_REFINER_BASE_SHA256`、`WCM_REFINER_TRAIN_SPLIT`、`WCM_REFINER_VAL_SPLIT` 指定输入；Windows到WSL的路径传递由包装器处理。使用包装器时split参数应为相对数据根的路径。

## 验证与证据边界

新测试：`tests/test_models/test_wcm_recurrent_refiner.py`。原WCM测试族也进行了回归。覆盖零初始化/零轮等价、共享参数、单次K/V、梯度、填充/移位、旧分类器复用、缓存读出、odd/whole/tensor/slide、严格权重加载、恢复和独立评估。

`tools/verify_refiner_checkpoint.py` 使用真实Mamba和旧DF权重，在合成输入上比较原基座与新增零初始化结构，并检查冻结梯度。输出JSON记录checkpoint哈希、后端、形状、差异、参数量及 `optimizer_steps=0`。这是权重与实现兼容性验证，不是训练收敛或分割精度结果。

旧域筛选通过后，才能在正式HardCrack同轨配置中确认效果。后续NF/U3/移位对照按拟保留的主张开展，不通过“差异不显著”自动宣称等效。
