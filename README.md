# Audio-Vibration Distillation Project

This repository contains the code, data organization, trained model outputs, and evaluation results for an audio-vibration knowledge distillation experiment. The task is to classify bottle/container conditions from synchronized audio and vibration recordings. The project compares three model types:

- an audio-only spectrogram CNN,
- a vibration-only TCN teacher model,
- an audio-only student model trained by distillation from the vibration teacher.

The final distilled model uses audio only at inference time, although vibration information is used during training through teacher soft labels.

## 1. Dataset

### Raw Synchronized Data

The raw audio and vibration recordings are stored in:

```text
sensor_plot_project/
```

This folder contains the original data collected during the experiments. Audio and vibration were collected during the same time periods, so each class has synchronized audio and vibration signals. During preprocessing, the raw vibration logs are converted into CSV files under `processed/`, and the training scripts align audio windows and vibration windows by time.

The main classes used in this project are:

| Class | Audio file | Raw vibration CSV |
| --- | --- | --- |
| `bottle_1000` | `sensor_plot_project/bottle_1/1.wav` | `sensor_plot_project/bottle_1/1.csv` |
| `bottle_500` | `sensor_plot_project/bottle_500/500ml.wav` | `sensor_plot_project/bottle_500/500ml.csv` |
| `bottle_can_330` | `sensor_plot_project/bottle_yi_330/yi_330.wav` | `sensor_plot_project/bottle_yi_330/yi_330.csv` |
| `bottle_can_500` | `sensor_plot_project/bottle_yi_500/yi_500.wav` | `sensor_plot_project/bottle_yi_500/yi_500.csv` |
| `error` | `sensor_plot_project/error/error.wav` | `sensor_plot_project/error/error.csv` |

The five main categories correspond to a 1 L bottle, a 500 mL bottle, a 330 mL can, a 500 mL can, and an error class. The `error` class contains audio and vibration recordings collected when the machine experienced abnormal behavior during bottle insertion, such as jamming, hesitation, or unusual mechanical noise.

The folder also contains additional data used for separate checks, such as:

- `sensor_plot_project/bottle_yi_20/20.wav`: 20 mL can audio test data.
- `sensor_plot_project/bottle_empty/empty.csv`: vibration data collected when the machine was running empty.

These additional files are retained in the repository but were not used in the main training, distillation, or evaluation experiments.

### Processed Vibration Data

The processed vibration CSV files are stored in:

```text
processed/
```

Main processed files:

- `processed/bottle_1000.csv`
- `processed/bottle_500.csv`
- `processed/bottle_can_330.csv`
- `processed/bottle_can_500.csv`
- `processed/error.csv`

Additional processed files are also retained:

- `processed/bottle_empty.csv`
- `processed/bottle_yi_330.csv`
- `processed/bottle_yi_500.csv`

The preprocessing notebook is:

```text
Step1-Log.ipynb
```

It converts the raw sensor logs into CSV files that can be used by the model training scripts. The default audio-vibration pairing is defined in `DEFAULT_PAIRED_CONFIG` inside `train_audio_student_distill.py`.

## 2. Model Types

### Audio-Only Model

The audio-only model uses only `.wav` audio files. The training script is:

```bash
python train_time_aligned_audio_spectrogram.py
```

The script resamples audio to 16 kHz, applies high-pass filtering, extracts sliding windows, converts each window to a log spectrogram, and trains a CNN classifier.

Audio-only model outputs:

- Original/main version: `original_time_aligned_audio_spectrogram_outputs/audio_spectrogram_cnn.keras`
- Newly trained version: `time_aligned_audio_spectrogram_outputs/audio_spectrogram_cnn.keras`

### Vibration-Only Model

The vibration-only model uses the processed vibration CSV files and serves as the teacher model during distillation. The training script is:

```bash
python train_time_aligned_tcn_teacher.py
```

The script uses the audio timeline to define aligned event windows, but the teacher model input is vibration only. The teacher architecture is a TCN-based vibration classifier.

Vibration teacher outputs:

- Original/main version: `original_time_aligned_tcn_teacher_outputs/teacher_vibration.keras`
- Newly trained version: `time_aligned_tcn_teacher_outputs/teacher_vibration.keras`

### Distilled Audio Student Model

The distilled model is an audio-only student. During training, it uses soft labels from the vibration teacher; during inference, it only requires audio input.

There are two important distillation scripts:

- `train_audio_student_distill_paper_kl_ce.py`: KL+CE distillation with teacher confidence weighting. The KL term is weighted by teacher confidence.
- `train_audio_student_distill_tcn_teacher.py`: no-confidence distillation. The KL term is not weighted by teacher confidence.

The main confidence-weighted distillation script is:

```bash
python train_audio_student_distill_paper_kl_ce.py
```

The distillation loss is:

```text
L = (1 - alpha) * CE + alpha * KL
```

where `CE` is the cross-entropy loss with ground-truth labels, and `KL` is the KL divergence between the teacher soft labels and the student outputs. The default temperature is `temperature=3.0`, and the default balance parameter is `alpha=0.4`.

The distilled student model is saved as:

```text
student_audio_only.keras
```

Although the student is trained with help from a vibration teacher, the saved student model is audio-only at inference time.

When running the distillation scripts, make sure to switch the data and model paths correctly, especially:

```text
--teacher-model
--student-init-model
```

For the original/main models, use:

```text
original_time_aligned_tcn_teacher_outputs/
original_time_aligned_audio_spectrogram_outputs/
```

For the newly trained models, use:

```text
time_aligned_tcn_teacher_outputs/
time_aligned_audio_spectrogram_outputs/
```

## 3. Important Experimental Note

Due to an operating system update during the project, the main reported evaluations were performed with the originally saved audio-only and vibration-only models:

- `original_time_aligned_audio_spectrogram_outputs/`
- `original_time_aligned_tcn_teacher_outputs/`

The corresponding distillation, alpha sweep, no-confidence, and noise robustness results are mainly stored in output folders with the `_old` suffix.

Additional distillation experiments were also performed with newly trained models:

- `time_aligned_audio_spectrogram_outputs/`
- `time_aligned_tcn_teacher_outputs/`

The alpha sweep results using the newly trained models are stored in:

```text
distill_outputs_paper_kl_ce_alpha_sweep_new/
```

For the main experimental results, prioritize the `original_*` model folders and the output folders ending in `_old`. The `time_aligned_*` and `*_new` folders are supplementary experiments using newly trained models.

## 4. Code Files

| File | Description |
| --- | --- |
| `Step1-Log.ipynb` | Preprocesses raw sensor logs and exports vibration CSV files to `processed/`. |
| `audio_student_model.py` | Audio utilities and CNN student definition, including audio loading, normalization, windowing, log-spectrogram computation, and audio CNN construction. |
| `event_windows.py` | Time-alignment and vibration-window utilities, including vibration CSV loading, synchronized time-range computation, and vibration-window resampling. |
| `layers/tcn.py` | TCN layer implementation used by the vibration teacher. |
| `train_time_aligned_audio_spectrogram.py` | Trains the time-aligned audio-only spectrogram CNN and saves `audio_spectrogram_cnn.keras`. |
| `train_time_aligned_tcn_teacher.py` | Trains the time-aligned vibration-only TCN teacher and saves `teacher_vibration.keras`. |
| `train_audio_student_distill_paper_kl_ce.py` | Main confidence-weighted KL+CE distillation script for training an audio-only student. Supports teacher confidence weighting, class weights, and focus-margin options. |
| `train_audio_student_distill_tcn_teacher.py` | No-confidence distillation script. The KL term is not weighted by teacher confidence. |
| `train_audio_student_distill_paper_kl_ce_noise.py` | Runs confidence-weighted distillation and waveform-level SNR noise evaluation. |
| `eval_no_confidence_student_waveform_noise.py` | Evaluates an existing no-confidence student under waveform-level SNR noise. |
| `run_paper_kl_ce_alpha_sweep_plain.py` | Runs multiple alpha values for KL+CE distillation and summarizes the results. |
| `train_audio_spectrogram.py` | Earlier audio-only spectrogram training script with support for window-size sweeps. |
| `train_audio_student_distill.py` | Earlier distillation script and shared utilities, including the default paired configuration, plotting, confusion-matrix saving, and alpha-sweep helpers. |
| `test.py` | Runs single-file inference with a trained `student_audio_only.keras` model and saves `test/result.json`. |

## 5. Recommended Workflow

### Step 1: Preprocess Raw Logs

Run:

```text
Step1-Log.ipynb
```

This produces processed vibration CSV files under `processed/`.

### Step 2: Train the Audio-Only Model

```bash
python train_time_aligned_audio_spectrogram.py \
  --output-dir time_aligned_audio_spectrogram_outputs
```

Main outputs:

- `audio_spectrogram_cnn.keras`
- `audio_spectrogram_classification_report.txt`
- `audio_spectrogram_confusion_matrix.png`
- `audio_spectrogram_accuracy_loss.png`
- `audio_spectrogram_meta.json`
- `dataset_stats.json`
- `metrics.json`

### Step 3: Train the Vibration Teacher

```bash
python train_time_aligned_tcn_teacher.py \
  --output-dir time_aligned_tcn_teacher_outputs
```

Main outputs:

- `teacher_vibration.keras`
- `teacher_classification_report.txt`
- `teacher_confusion_matrix.png`
- `teacher_accuracy_loss.png`
- `teacher_run_meta.json`
- `window_dataset_stats.json`

### Step 4: Run Main Distillation with Original Models

This is the confidence-weighted distillation experiment using `train_audio_student_distill_paper_kl_ce.py`.

```bash
python train_audio_student_distill_paper_kl_ce.py \
  --teacher-model original_time_aligned_tcn_teacher_outputs/teacher_vibration.keras \
  --student-init-model original_time_aligned_audio_spectrogram_outputs/audio_spectrogram_cnn.keras \
  --output-dir distill_outputs_paper_kl_ce_old
```

Main outputs:

- `student_audio_only.keras`
- `student_classification_report.txt`
- `student_confusion_matrix.png`
- `student_accuracy_loss.png`
- `student_ce_kl.png`
- `teacher_classification_report.txt`
- `teacher_confusion_matrix.png`
- `distill_run_meta.json`
- `paired_dataset_stats.json`
- `label_to_int.json`

### Step 5: Run Alpha Sweep

Alpha sweep with original/main models:

```bash
python run_paper_kl_ce_alpha_sweep_plain.py \
  --output-dir distill_outputs_paper_kl_ce_alpha_sweep_plain_old
```

Example alpha run with newly trained models:

```bash
python train_audio_student_distill_paper_kl_ce.py \
  --teacher-model time_aligned_tcn_teacher_outputs/teacher_vibration.keras \
  --student-init-model time_aligned_audio_spectrogram_outputs/audio_spectrogram_cnn.keras \
  --alpha 0.4 \
  --output-dir distill_outputs_paper_kl_ce_alpha_sweep_new/alpha_0_40
```

To sweep multiple alpha values with the newly trained models, repeat the command above with different `--alpha` and `--output-dir` values, or extend `run_paper_kl_ce_alpha_sweep_plain.py` so that it also passes `--teacher-model` and `--student-init-model`.

Alpha sweep outputs:

- `alpha_0_10/`, `alpha_0_20/`, etc.: full result folders for each alpha value.
- `alpha_sweep_summary.csv`
- `alpha_sweep_accuracy.png`
- `best_alpha.json`

### Step 6: Run No-Confidence Distillation

This experiment does not use teacher confidence weighting. It uses `train_audio_student_distill_tcn_teacher.py`.

```bash
python train_audio_student_distill_tcn_teacher.py \
  --teacher-model original_time_aligned_tcn_teacher_outputs/teacher_vibration.keras \
  --student-init-model original_time_aligned_audio_spectrogram_outputs/audio_spectrogram_cnn.keras \
  --output-dir distill_outputs_paper_kl_ce_no_confidence_old
```

The no-confidence setting means that the KL term is not weighted by teacher confidence. This result is used for comparison against the confidence-weighted distilled model.

### Step 7: Run Noise Robustness Evaluation

Noise evaluation for the confidence-weighted distilled model:

```bash
python train_audio_student_distill_paper_kl_ce_noise.py \
  --teacher-model original_time_aligned_tcn_teacher_outputs/teacher_vibration.keras \
  --student-init-model original_time_aligned_audio_spectrogram_outputs/audio_spectrogram_cnn.keras \
  --output-dir distill_outputs_paper_kl_ce_waveform_noise_snr_test_old \
  --noise-snr-db-list 30 20 10 5 0
```

Noise evaluation for the no-confidence student:

```bash
python eval_no_confidence_student_waveform_noise.py \
  --student-model distill_outputs_paper_kl_ce_no_confidence_old/student_audio_only.keras \
  --output-dir distill_outputs_paper_kl_ce_no_confidence_waveform_noise_snr_test_old \
  --noise-snr-db-list 30 20 10 5 0
```

Noise evaluation outputs:

- `waveform_noise_snr_eval/noise_eval_summary.csv`
- `waveform_noise_snr_eval/noise_eval_summary.json`
- `waveform_noise_snr_eval/snr_30_0db/`
- `waveform_noise_snr_eval/snr_20_0db/`
- `waveform_noise_snr_eval/snr_10_0db/`
- `waveform_noise_snr_eval/snr_5_0db/`
- `waveform_noise_snr_eval/snr_0_0db/`

Each SNR subfolder contains:

- `student_classification_report.txt`
- `student_confusion_matrix.png`
- `noise_eval_meta.json`

## 6. Result Directories

### Original Audio-Only Model

```text
original_time_aligned_audio_spectrogram_outputs/
```

Important files:

- `audio_spectrogram_cnn.keras`: audio-only CNN model.
- `metrics.json`: test-set metrics.
- `audio_spectrogram_classification_report.txt`: classification report.
- `audio_spectrogram_confusion_matrix.png`: confusion matrix.
- `dataset_stats.json`: per-class window counts and time-alignment statistics.

### Original Vibration Teacher

```text
original_time_aligned_tcn_teacher_outputs/
```

Important files:

- `teacher_vibration.keras`: vibration-only TCN teacher.
- `teacher_run_meta.json`: training settings and test metrics.
- `teacher_classification_report.txt`: classification report.
- `teacher_confusion_matrix.png`: confusion matrix.
- `window_dataset_stats.json`: per-class vibration-window statistics.

### Newly Trained Audio and Vibration Models

```text
time_aligned_audio_spectrogram_outputs/
time_aligned_tcn_teacher_outputs/
```

These folders follow the same structure as the `original_*` folders and are used for supplementary experiments after the system update.

### Main Distillation Results

- `distill_outputs_paper_kl_ce_old/`: original models with confidence-weighted KL+CE distillation.
- `distill_outputs_paper_kl_ce_no_confidence_old/`: original models with no-confidence distillation.
- `distill_outputs_paper_kl_ce_alpha_sweep_plain_old/`: alpha sweep using the original models.
- `distill_outputs_paper_kl_ce_waveform_noise_snr_test_old/`: waveform-level SNR noise evaluation for the confidence-weighted distilled model.
- `distill_outputs_paper_kl_ce_no_confidence_waveform_noise_snr_test_old/`: waveform-level SNR noise evaluation for the no-confidence student.

### Supplementary Distillation with Newly Trained Models

```text
distill_outputs_paper_kl_ce_alpha_sweep_new/
```

The metadata in this folder indicates that the teacher is loaded from `time_aligned_tcn_teacher_outputs/teacher_vibration.keras`, and the initial student is loaded from `time_aligned_audio_spectrogram_outputs/audio_spectrogram_cnn.keras`.

## 7. Single-File Inference

After training a distilled student, use `test.py` to classify a single `.wav` file:

```bash
python test.py sensor_plot_project/bottle_500/500ml.wav \
  --run-dir distill_outputs_paper_kl_ce_old \
  --output-dir test
```

Outputs:

- Predicted label and class probabilities printed in the terminal.
- `test/result.json`

## 8. Recommended Files for Public Release

Recommended files and folders to include:

- Code: all `.py` files, `layers/`, and `Step1-Log.ipynb`.
- Raw synchronized data: `sensor_plot_project/`.
- Processed vibration data: `processed/`.
- Original/main model outputs: `original_time_aligned_audio_spectrogram_outputs/` and `original_time_aligned_tcn_teacher_outputs/`.
- Supplementary newly trained model outputs: `time_aligned_audio_spectrogram_outputs/` and `time_aligned_tcn_teacher_outputs/`.
- Distillation results: `distill_outputs_paper_kl_ce_old/`, `distill_outputs_paper_kl_ce_no_confidence_old/`, `distill_outputs_paper_kl_ce_alpha_sweep_plain_old/`, `distill_outputs_paper_kl_ce_waveform_noise_snr_test_old/`, and `distill_outputs_paper_kl_ce_no_confidence_waveform_noise_snr_test_old/`.
- Supplementary new-model distillation results: `distill_outputs_paper_kl_ce_alpha_sweep_new/`.

The following files and folders do not need to be included:

- `__pycache__/`
- `.DS_Store`
- `.mplconfig/`

