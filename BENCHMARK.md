# SPNet-Large：NYU 随机权重纯推理测速

从仓库根目录执行，使用独立环境，不运行原训练/测试入口，也不下载权重：

```bash
conda env create -f environment.yaml
conda activate SPNet
mkdir -p data
ln -s ../../data/nyudepthv2_h5 data/nyudepthv2_h5
python -m pytest tests/test_inference.py -q
python scripts/benchmark_nyu.py --output outputs/benchmark_nyu_new_run
```

已有环境/链接时不要重复创建；输出目录必须尚不存在。搬到其他目录时只调整数据软链接；`RGBD_Datasets/`、`Hole_Datasets/`、`Test_Datasets/`、`checkpoints/` 原内容保持不动。

## 模型和显式输入适配

- 保留 `config.py` / `test.py` 的默认 **Large/CNX**：channels `[192,384,768,1536]`，blocks `[3,3,27,3]`，drop-path=0.2（eval 关闭随机丢弃），作者 Xavier 初始化，不改为 Tiny。
- 外部统一 NYU 228×304、500 点、seed=2023、ImageNet RGB 归一化及确定性采样。原项目 RGB 为 [0,1]；这里使用统一 RGB，是随机结构效率测量，不宣称原论文精度复现。
- 保留 5 通道输入：RGB、depth/20（室内深度编码）、validity mask（有稀疏点=1）。wrapper 内 mask、深度缩放、右补 16/下补 28 行零至 **256×320**、预测裁剪及换回米均计入前向。内部尺寸满足 5 次下采样，不改 encoder/decoder 结构。

## 口径与记录

RTX 4060 Ti、FP32、batch=1，TF32/autocast/compile/CUDA graphs 关闭，cuDNN benchmark=False、deterministic=False，CPU threads=1；`eval()` + `inference_mode()`。样本索引 0、326、653 各预热 100 次，CUDA events 逐帧同步计时 1000 次，合并平均/P95；FPS=1000/平均毫秒，另记录同步墙钟时间。

输入预放 GPU，排除加载/H2D、真值、损失、指标、日志、可视化；显存另测单次前向峰值 allocated，含模型与输入。参数含冻结参数，共享参数只计一次；FP32 state_dict 文件含 buffers 和序列化开销。MACs 为实际内部尺寸上的卷积/转置卷积累加，不含 SP-Norm 的 std_mean/归约、逐元素乘法、mask/补边等；不是完整 FLOPs。

每次保存 JSON、逐次计时/GPU 状态、配置、随机 state_dict、源码哈希/完整快照、Git 差异和独立环境清单；三个真实样本须与完整原前向 allclose(rtol=1e-5, atol=1e-6)。随机权重不报告 RMSE，也不代表训练后 checkpoint 延迟。短验证可加 `--warmup 2 --iterations 5 --groups 1`。
