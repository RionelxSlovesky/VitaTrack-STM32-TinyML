# VitaTrack WESAD stress model

This folder contains the offline ML pipeline for VitaTrack.

## Expected WESAD folder structure

WESAD/
  S2/S2.pkl
  S3/S3.pkl
  ...
  S17/S17.pkl

## Install dependencies

```bash
python -m pip install numpy pandas scipy scikit-learn tensorflow neurokit2
```

## 1. Extract 60-second ECG HRV windows

```bash
python extract_wesad_hrv.py --wesad C:\path\to\WESAD --out wesad_hrv_features.csv
```

Binary target:
- 1 = stress
- 0 = non-stress (baseline or amusement)

Features:
- mean RR
- mean heart rate
- SDNN
- RMSSD
- pNN50
- RR range

## 2. Train subject-independent TinyML classifier

```bash
python train_stress_model.py --csv wesad_hrv_features.csv --outdir stress_model
```

Architecture:
6 -> Dense(12, ReLU) -> Dense(8, ReLU) -> Dense(1, sigmoid)

The script:
- performs subject-wise train/validation/test splits,
- normalizes using training subjects only,
- balances training classes,
- uses early stopping,
- reports test metrics,
- exports a float Keras model,
- exports a fully-INT8 TFLite model,
- writes feature normalization constants to a C header.

## 3. Outputs

`stress_model/`
- stress_float.keras
- stress_int8.tflite
- feature_mean.npy
- feature_std.npy
- feature_order.json
- metrics.json
- vitatrack_stress_model_metadata.h
