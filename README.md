# PlenopticDreamer 复现实验代码

独立的研究代码仓库，包含 PlenopticDreamer 相关的公开相机模块和我们的训练、数据处理、推理及评测实现。不是作者官方实现，也不包含作者微调权重。

## 代码

- `upstream_camera/cosmos_transfer2/_src/predict2/camera/`：NVIDIA 公开的相机网络、相机射线工具及相关模型配置，保留原始版权声明。
- `scripts/plenoptic/`：模型加载、参考与目标布局、flow matching loss、采样及共用数据模块。
- `scripts/plenoptic432_a_continue/`：432×768 训练、数据缓存、相机条件、单参考续训、相机结构消融及评测。当前 A 组入口为 `overnight_camera_abc.py`，调度入口为 `continue_supervisor.py`。
- `scripts/plenoptic_recovery/`：参考视频描述的解析、校验和重试。
- `configs/a_continue_batch40.json`：当前 A 组训练配置。

## 依赖

依赖 [NVIDIA Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)，使用的 upstream 版本为 `2ff49d0717af02057ae79bc75c00fbff9da1b4e7`。`upstream_camera` 是该版本相关源码的原样摘录，供查看相机网络；不是完整的 Cosmos Python 包，单独加入 PYTHONPATH 不能替代安装完整依赖。

训练脚本保留了实验时的本机路径和已有 manifest/checkpoint 协议。迁移运行需要配置路径、依赖、数据 manifest 及合法取得的模型权重，当前代码不是开箱即用的独立训练包。

仓库不包含数据集、缓存、checkpoint 或模型权重。NVIDIA 源码适用 Apache-2.0，许可证见 `LICENSE-COSMOS-APACHE-2.0`；其他依赖和权重遵守各自的许可。
