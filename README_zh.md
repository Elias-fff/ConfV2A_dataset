# ConfV2A 数据集与代码

[English](README.md)

**面向回收压缩机瓶罐分类的置信度引导振动到声学蒸馏**

ConfV2A 利用同步振动信号辅助训练纯音频瓶罐分类器。训练阶段使用振动教师模型，而部署时只需要麦克风音频。

![ConfV2A 概览](figures/confv2a_overview.png)

## 项目亮点

- **纯音频部署：** 推理阶段不需要振动传感器。
- **振动引导训练：** 使用振动 TCN 教师模型监督声学 CNN 学生模型。
- **置信度感知蒸馏：** 教师模型预测不确定时，对 KL 蒸馏项的贡献更小。
- **真实压缩机数据：** 在瓶罐压缩过程中采集同步音频和振动信号。
- **可复现实验输出：** 仓库包含训练模型、报告、混淆矩阵、Alpha 扫描和噪声鲁棒性结果。

## 如何阅读本仓库

如果你是第一次访问本项目，建议按照以下顺序阅读：

1. **理解核心思路：** 首先查看上方的整体框架图。
2. **了解数据采集设置：** 阅读[数据集概览](#数据集概览)。
3. **了解模型流程：** 阅读[方法](#方法)。
4. **按照代码顺序阅读：** 查看[代码导航](#代码导航)。
5. **查看实验输出和对比：** 阅读[实验结果](#实验结果)和[仓库结构](#仓库结构)。

## 主要结果

在自行采集的瓶罐压缩数据集上，ConfV2A 将纯音频基线的测试准确率从 **79.01%** 提高到 **90.24%**，同时在推理阶段仍然只使用音频。

| 方法 | 推理输入 | 准确率 | Macro-F1 |
| --- | --- | ---: | ---: |
| 纯音频基线 | 音频 | 79.01% | 78.19% |
| 振动教师模型 | 振动 | 91.14% | 91.80% |
| 标准 KD 音频学生模型 | 音频 | 86.99% | 86.62% |
| **ConfV2A 音频学生模型** | **音频** | **90.24%** | **89.83%** |

## 数据集概览

本项目的自采数据集包含来自瓶罐回收压缩机的同步麦克风音频和振动记录。音频窗口和振动窗口在时间上对齐，使振动教师模型能够在训练阶段监督声学学生模型。

<p align="center">
  <img src="figures/compaction_line.png" alt="瓶罐压缩生产线" width="38%">
  <img src="figures/sensor_placement.png" alt="压缩机中的传感器位置" width="58%">
</p>

| 项目 | 说明 |
| --- | --- |
| 任务 | 回收压缩机中的瓶罐分类 |
| 模态 | 麦克风音频 + 振动传感器 |
| 类别 | 1 L 瓶、500 mL 瓶、330 mL 罐、500 mL 罐、错误类 |
| 事件/窗口 | 检测到 120 个事件，保留 1206 个窗口 |
| 窗口长度 | 0.8 秒对齐音频—振动窗口 |
| 数据划分 | 按时间分块的 70% / 15% / 15% 划分 |
| 原始数据 | `data/raw/sensor_plot_project/` |
| 处理后的振动数据 | `data/processed/` |

主要类别文件：

| 类别 | 音频文件 | 原始振动 CSV |
| --- | --- | --- |
| `bottle_1000` | `data/raw/sensor_plot_project/bottle_1/1.wav` | `data/raw/sensor_plot_project/bottle_1/1.csv` |
| `bottle_500` | `data/raw/sensor_plot_project/bottle_500/500ml.wav` | `data/raw/sensor_plot_project/bottle_500/500ml.csv` |
| `bottle_can_330` | `data/raw/sensor_plot_project/bottle_yi_330/yi_330.wav` | `data/raw/sensor_plot_project/bottle_yi_330/yi_330.csv` |
| `bottle_can_500` | `data/raw/sensor_plot_project/bottle_yi_500/yi_500.wav` | `data/raw/sensor_plot_project/bottle_yi_500/yi_500.csv` |
| `error` | `data/raw/sensor_plot_project/error/error.wav` | `data/raw/sensor_plot_project/error/error.csv` |

`error` 类包含异常投放或机器异常行为，例如卡住、迟滞或异常机械噪声。`bottle_yi_20/` 和 `bottle_empty/` 等额外文件仅用于检查，不属于主要训练和评估设置。

## 方法

ConfV2A 将振动信号作为仅在训练阶段使用的特权模态：

1. 训练纯音频频谱 CNN 基线。
2. 使用对齐的振动窗口训练纯振动 TCN 教师模型。
3. 利用真实标签和教师软标签，将振动教师模型的知识蒸馏到声学学生模型。
4. 根据教师模型的置信度对 KL 项进行加权，降低不可靠教师预测的影响。
5. 部署时只使用训练完成的音频学生模型。

主要损失函数为：

```text
L = (1 - alpha) * CE + alpha * c_teacher * KL
```

其中，`CE` 是基于真实标签的交叉熵损失，`KL` 用于传递教师软标签信息，`c_teacher` 是教师模型置信度。论文主要实验采用 `temperature=3.0` 和 `alpha=0.4`。

## 代码导航

源代码按照用途整理在 `src/` 中。阅读或复现实验时，建议按照以下顺序：

> [!IMPORTANT]
> `src/` 下的子文件夹用于在 GitHub 上清晰展示项目结构。当前脚本保留了原有的本地导入方式，因此不能直接在该分类结构下运行。
>
> 运行代码时，请将 `src/models/`、`src/utils/`、`src/training/`、`src/experiments/` 和 `src/evaluation/` 中的 Python 源文件集中放入同一个工作目录，并在该目录下保留 `layers/tcn.py`。也就是说，实际执行时应使用扁平化源代码结构，而不是仓库网页展示的分类结构。

| 步骤 | 文件 | 用途 |
| --- | --- | --- |
| 1 | `src/preprocessing/Step1-Log.ipynb` | 将原始传感器日志转换为 `data/processed/` 下的振动 CSV 文件 |
| 2 | `src/training/train_time_aligned_audio_spectrogram.py` | 训练纯音频频谱 CNN 基线 |
| 3 | `src/training/train_time_aligned_tcn_teacher.py` | 训练纯振动 TCN 教师模型 |
| 4 | `src/training/train_audio_student_distill_paper_kl_ce.py` | 使用置信度加权 KL + CE 训练主要 ConfV2A 音频学生模型 |
| 5 | `src/training/train_audio_student_distill_tcn_teacher.py` | 训练不使用置信度加权的标准 KD 基线 |
| 6 | `src/experiments/train_audio_student_distill_paper_kl_ce_noise.py` | 运行 ConfV2A 噪声鲁棒性实验 |
| 7 | `src/evaluation/eval_no_confidence_student_waveform_noise.py` | 运行标准 KD 噪声鲁棒性评估 |
| 8 | `src/experiments/run_paper_kl_ce_alpha_sweep_plain.py` | 运行 Alpha 扫描实验 |
| 9 | `src/evaluation/test.py` | 使用已训练的学生模型对单个音频文件进行推理 |

准备好扁平化工作目录后，可以按照以下顺序运行：

```bash
python train_time_aligned_audio_spectrogram.py
python train_time_aligned_tcn_teacher.py
python train_audio_student_distill_paper_kl_ce.py
```

## 实验结果

### 噪声鲁棒性

在波形级高斯噪声条件下，ConfV2A 相对于标准 KD 始终保持优势。

| 模型 | 无噪声 | 30 dB | 20 dB | 10 dB | 5 dB | 0 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 标准 KD | 86.99% | 87.80% | 88.62% | 83.74% | 74.80% | 55.28% |
| **ConfV2A** | **90.24%** | **91.87%** | **90.24%** | **86.18%** | **78.86%** | **56.10%** |

### 跨数据集比较

同一框架还在三个公开的同步音频—振动数据集上进行了评估。

| 数据集 | 音频基线 | 振动教师模型 | ConfV2A |
| --- | ---: | ---: | ---: |
| 自采瓶罐数据集 | 79.01% | 91.14% | **90.24%** |
| UOEMD | 83.33% | 97.45% | **97.96%** |
| MaFaulDa | 86.61% | 89.56% | **89.95%** |
| QU-DMBF | 99.63% | 88.54% | **99.92%** |

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| `data/raw/` | 原始音频和振动记录 |
| `data/processed/` | 根据传感器日志生成的振动 CSV 文件 |
| `src/models/` | 音频学生模型和 TCN 模型组件 |
| `src/preprocessing/` | 传感器日志预处理 Notebook |
| `src/training/` | 基线、教师模型和学生模型训练脚本 |
| `src/experiments/` | 噪声鲁棒性和 Alpha 扫描实验 |
| `src/evaluation/` | 评估和单文件推理脚本 |
| `src/utils/` | 共享事件窗口和同步工具 |
| `results/main/` | 论文主要模型和报告 |
| `results/baselines/` | 标准知识蒸馏基线 |
| `results/ablations/` | Alpha 扫描和消融实验输出 |
| `results/robustness/` | 波形噪声鲁棒性评估 |
| `results/supplementary/` | 重新训练和补充实验 |
| `results/inference/` | 单文件推理输出 |
| `figures/` | README 和项目网页使用的图片 |
| `index.html` | 可选的轻量 GitHub Pages 项目网页 |

<details>
<summary>详细文件夹内容</summary>

| 路径 | 用途 |
| --- | --- |
| `data/raw/sensor_plot_project/` | 原始同步音频和振动记录 |
| `data/processed/` | 处理后的振动 CSV 文件 |
| `src/preprocessing/Step1-Log.ipynb` | 传感器日志预处理 Notebook |
| `src/models/audio_student_model.py` | 音频加载、切窗、频谱和 CNN 工具 |
| `src/utils/event_windows.py` | 时间对齐音频—振动窗口工具 |
| `src/models/layers/tcn.py` | 振动教师模型使用的 TCN 层 |
| `results/main/original_time_aligned_audio_spectrogram_outputs/` | 主要结果使用的原始纯音频 CNN 模型和报告 |
| `results/main/original_time_aligned_tcn_teacher_outputs/` | 主要结果使用的原始振动教师模型和报告 |
| `results/supplementary/time_aligned_audio_spectrogram_outputs/` | 新训练的纯音频模型输出 |
| `results/supplementary/time_aligned_tcn_teacher_outputs/` | 新训练的振动教师模型输出 |
| `results/main/distill_outputs_paper_kl_ce_old/` | 使用原始模型得到的主要 ConfV2A 结果 |
| `results/baselines/distill_outputs_paper_kl_ce_no_confidence_old/` | 使用原始模型得到的标准 KD 结果 |
| `results/ablations/distill_outputs_paper_kl_ce_alpha_sweep_plain_old/` | 使用原始模型进行的 Alpha 扫描 |
| `results/robustness/distill_outputs_paper_kl_ce_waveform_noise_snr_test_old/` | ConfV2A 波形级噪声评估 |
| `results/robustness/distill_outputs_paper_kl_ce_no_confidence_waveform_noise_snr_test_old/` | 标准 KD 波形级噪声评估 |
| `results/supplementary/distill_outputs_paper_kl_ce_alpha_sweep_new/` | 使用新训练模型进行的补充 Alpha 扫描 |

</details>

## 实验说明

论文报告的主要评估使用以下目录中的模型：

```text
results/main/original_time_aligned_audio_spectrogram_outputs/
results/main/original_time_aligned_tcn_teacher_outputs/
```

对于与论文主要实验对应的结果，请优先查看名称以 `_old` 结尾的文件夹。`time_aligned_*` 和 `*_new` 文件夹属于系统更新后使用重新训练模型获得的补充结果。

## 联系方式

Junfu Zhang  
Friedrich-Alexander-Universität Erlangen-Nürnberg  
junfu.zhang@fau.de
