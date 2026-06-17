# ConfV2A Dataset and Code

**Confidence-Guided Vibration-to-Acoustic Distillation for Bottle Classification in Recycling Compactors**

ConfV2A trains an audio-only bottle classifier with help from synchronized vibration signals. A vibration teacher is used during training, but deployment only needs microphone audio.

![ConfV2A overview](figures/overview.png)

## Highlights

- **Audio-only deployment:** no vibration sensors are required at inference time.
- **Vibration-guided training:** a TCN vibration teacher supervises a CNN acoustic student.
- **Confidence-aware distillation:** uncertain teacher predictions contribute less to the KL distillation term.
- **Real compactor data:** synchronized audio and vibration were collected during bottle/can compression events.
- **Reproducible outputs:** trained models, reports, confusion matrices, alpha sweeps, and noise robustness results are included.

## How To Read This Repository

If you are visiting this project for the first time, follow this order:

1. **Understand the idea:** start with the overview figure above.
2. **Inspect the data collection setup:** see [Dataset Overview](#dataset-overview).
3. **Read the model pipeline:** see [Method](#method).
4. **Follow the code order:** see [Code Navigation](#code-navigation).
5. **Check outputs and comparisons:** see [Results](#results) and [Repository Map](#repository-map).

## Key Result

On the self-collected bottle/can compaction dataset, ConfV2A improves the audio-only baseline from **79.01%** to **90.24%** test accuracy while still using only audio at inference.

| Method | Inference input | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| Audio-only baseline | Audio | 79.01% | 78.19% |
| Vibration teacher | Vibration | 91.14% | 91.80% |
| Standard KD audio student | Audio | 86.99% | 86.62% |
| **ConfV2A audio student** | **Audio** | **90.24%** | **89.83%** |

## Dataset Overview

The self-collected dataset contains synchronized microphone audio and vibration recordings from a bottle-recycling compactor. Audio and vibration windows are temporally aligned so that the vibration teacher can supervise the acoustic student during training.

<p align="center">
  <img src="figures/compaction_line.png" alt="Bottle compaction line" width="38%">
  <img src="figures/sensor_placement.png" alt="Sensor placement in the compactor" width="58%">
</p>

| Item | Description |
| --- | --- |
| Task | Bottle/can classification in a recycling compactor |
| Modalities | Microphone audio + vibration sensors |
| Classes | 1 L bottle, 500 mL bottle, 330 mL can, 500 mL can, error |
| Events / windows | 120 detected events / 1206 retained windows |
| Window length | 0.8 s aligned audio-vibration windows |
| Split | 70% / 15% / 15% time-wise blocked split |
| Raw data | `data/raw/sensor_plot_project/` |
| Processed vibration data | `data/processed/` |

Main class files:

| Class | Audio file | Raw vibration CSV |
| --- | --- | --- |
| `bottle_1000` | `data/raw/sensor_plot_project/bottle_1/1.wav` | `data/raw/sensor_plot_project/bottle_1/1.csv` |
| `bottle_500` | `data/raw/sensor_plot_project/bottle_500/500ml.wav` | `data/raw/sensor_plot_project/bottle_500/500ml.csv` |
| `bottle_can_330` | `data/raw/sensor_plot_project/bottle_yi_330/yi_330.wav` | `data/raw/sensor_plot_project/bottle_yi_330/yi_330.csv` |
| `bottle_can_500` | `data/raw/sensor_plot_project/bottle_yi_500/yi_500.wav` | `data/raw/sensor_plot_project/bottle_yi_500/yi_500.csv` |
| `error` | `data/raw/sensor_plot_project/error/error.wav` | `data/raw/sensor_plot_project/error/error.csv` |

The `error` class contains abnormal insertion or machine behavior, such as jamming, hesitation, or unusual mechanical noise. Extra files such as `bottle_yi_20/` and `bottle_empty/` are retained for checks but are not part of the main training and evaluation setup.

## Method

ConfV2A uses vibration as a privileged training-time modality:

1. Train an audio-only spectrogram CNN baseline.
2. Train a vibration-only TCN teacher on aligned vibration windows.
3. Distill the vibration teacher into the acoustic student with ground-truth labels and teacher soft labels.
4. Weight the KL term by teacher confidence so unreliable teacher predictions have less influence.
5. Deploy only the trained audio student.

The main loss is:

```text
L = (1 - alpha) * CE + alpha * c_i * KL
```

where `CE` is the ground-truth cross-entropy loss, `KL` transfers teacher soft-label information, `c_i` is the teacher confidence, `temperature=3.0`, and `alpha=0.4` in the main experiment.

## Code Navigation

The code is grouped under `code/`. For reading or reproducing the pipeline, follow this order:

| Step | File | Purpose |
| --- | --- | --- |
| 1 | `code/Step1-Log.ipynb` | Converts raw sensor logs into processed vibration CSV files under `data/processed/`. |
| 2 | `code/train_time_aligned_audio_spectrogram.py` | Trains the audio-only spectrogram CNN baseline. |
| 3 | `code/train_time_aligned_tcn_teacher.py` | Trains the vibration-only TCN teacher. |
| 4 | `code/train_audio_student_distill_paper_kl_ce.py` | Trains the main ConfV2A audio student with confidence-weighted KL + CE. |
| 5 | `code/train_audio_student_distill_tcn_teacher.py` | Trains the standard KD baseline without confidence weighting. |
| 6 | `code/train_audio_student_distill_paper_kl_ce_noise.py` | Runs ConfV2A noise robustness evaluation. |
| 7 | `code/eval_no_confidence_student_waveform_noise.py` | Runs standard KD noise robustness evaluation. |
| 8 | `code/run_paper_kl_ce_alpha_sweep_plain.py` | Runs the alpha sweep experiment. |
| 9 | `code/test.py` | Runs single-file audio inference with a trained student model. |

Example command order:

```bash
python code/train_time_aligned_audio_spectrogram.py
python code/train_time_aligned_tcn_teacher.py
python code/train_audio_student_distill_paper_kl_ce.py
```

## Results

### Cross-Dataset Comparison

The same framework was also evaluated on three public synchronized audio-vibration datasets.

| Dataset | Audio baseline | Vibration teacher | ConfV2A |
| --- | ---: | ---: | ---: |
| Our bottle/can dataset | 79.01% | 91.14% | **90.24%** |
| UOEMD | 83.33% | 97.45% | **97.96%** |
| MaFaulDa | 86.61% | 89.56% | **89.95%** |
| QU-DMBF | 99.63% | 88.54% | **99.92%** |

## Repository Map

| Path | Purpose |
| --- | --- |
| `data/raw/` | Raw audio and vibration recordings |
| `data/processed/` | Processed vibration CSV files generated from sensor logs |
| `code/` | Training, evaluation, preprocessing, and inference code |
| `results/` | Trained models, reports, confusion matrices, alpha sweeps, and noise tests |
| `figures/` | Images used by this README and the project page |
| `index.html` | Optional lightweight GitHub Pages project page |

<details>
<summary>Detailed folder contents</summary>

| Path | Purpose |
| --- | --- |
| `data/raw/sensor_plot_project/` | Raw synchronized audio and vibration recordings |
| `data/processed/` | Processed vibration CSV files |
| `code/Step1-Log.ipynb` | Sensor-log preprocessing notebook |
| `code/audio_student_model.py` | Audio loading, windowing, spectrogram, and CNN utilities |
| `code/event_windows.py` | Time-aligned audio-vibration window utilities |
| `code/layers/tcn.py` | TCN layer used by the vibration teacher |
| `results/original_time_aligned_audio_spectrogram_outputs/` | Original audio-only CNN model and reports used for main results |
| `results/original_time_aligned_tcn_teacher_outputs/` | Original vibration teacher model and reports used for main results |
| `results/time_aligned_audio_spectrogram_outputs/` | Newly trained audio-only model outputs |
| `results/time_aligned_tcn_teacher_outputs/` | Newly trained vibration teacher outputs |
| `results/distill_outputs_paper_kl_ce_old/` | Main ConfV2A results with original models |
| `results/distill_outputs_paper_kl_ce_no_confidence_old/` | Standard KD results with original models |
| `results/distill_outputs_paper_kl_ce_alpha_sweep_plain_old/` | Alpha sweep with original models |
| `results/distill_outputs_paper_kl_ce_waveform_noise_snr_test_old/` | ConfV2A waveform-level noise evaluation |
| `results/distill_outputs_paper_kl_ce_no_confidence_waveform_noise_snr_test_old/` | Standard KD waveform-level noise evaluation |
| `results/distill_outputs_paper_kl_ce_alpha_sweep_new/` | Supplementary alpha sweep with newly trained models |

</details>

## Experimental Note

The main reported evaluations were performed with:

```text
results/original_time_aligned_audio_spectrogram_outputs/
results/original_time_aligned_tcn_teacher_outputs/
```

For the main paper-aligned results, prioritize folders ending in `_old`. The `time_aligned_*` and `*_new` folders are supplementary runs using newly trained models after a system update.

