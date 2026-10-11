# VitaTrack ECG replay validation

This package validates the same lightweight R-peak logic used in the STM32 firmware against annotated ECG from the PhysioNet MIT-BIH Arrhythmia Database.

## Install

```bash
python -m pip install numpy scipy wfdb
```

## Validate the detector

```bash
python validate_mitbih.py --record 100 --seconds 120
```

The script:
1. downloads MIT-BIH record 100,
2. resamples the first channel from 360 Hz to 250 Hz,
3. converts it to a plausible 12-bit ADC stream,
4. runs the VitaTrack filter + R-peak detector,
5. compares detections against MIT-BIH beat annotations.

## Export replay data for STM32 firmware

```bash
python export_replay_header.py --record 100 --start 5 --seconds 20 --out ecg_replay.h
```

Copy the resulting `ecg_replay.h` into the VitaTrack project and use it as a firmware replay source before physical hardware is available.

MIT-BIH is used only for engineering validation here. VitaTrack is not a diagnostic medical device.
