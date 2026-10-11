import argparse
import math
import numpy as np
import wfdb
from scipy.signal import resample_poly

TARGET_FS = 250
ADC_MIDPOINT = 2048.0
ADC_COUNTS_PER_MV = 400.0
REFRACTORY_SAMPLES = 63

BEAT_SYMBOLS = {
    "N", "L", "R", "B",
    "A", "a", "J", "S",
    "V", "r", "F",
    "e", "j", "n", "E",
    "/", "f", "Q", "?"
}

class VitaTrackDetector:
    def __init__(self, fs=250):
        self.fs = fs

        # Match the STM32 C state variables
        self.x_prev = 0.0
        self.hp_prev = 0.0
        self.lp_prev = 0.0

        self.filtered_prev = 0.0
        self.energy_floor = 1.0
        self.energy_peak = 1.0

        self.sample_index = 0
        self.last_rpeak_index = 0
        self.rpeak_count = 0

        self.detected = []

    def filter_sample(self, raw_sample):
        x = float(raw_sample) - ADC_MIDPOINT

        hp_alpha = 0.9876
        hp = hp_alpha * (self.hp_prev + x - self.x_prev)

        self.x_prev = x
        self.hp_prev = hp

        lp_alpha = 0.4680
        self.lp_prev = self.lp_prev + lp_alpha * (hp - self.lp_prev)

        return self.lp_prev

    def detect_rpeak(self, filtered_sample):
        derivative = filtered_sample - self.filtered_prev
        self.filtered_prev = filtered_sample

        energy = derivative * derivative

        self.energy_floor = (
            0.995 * self.energy_floor +
            0.005 * energy
        )

        if energy > self.energy_peak:
            self.energy_peak = energy
        else:
            self.energy_peak = (
                0.999 * self.energy_peak +
                0.001 * self.energy_floor
            )

        threshold = (
            self.energy_floor +
            0.30 * (self.energy_peak - self.energy_floor)
        )

        if self.sample_index > 2 * self.fs:
            samples_since_peak = self.sample_index - self.last_rpeak_index

            if (
                energy > threshold
                and (
                    self.rpeak_count == 0
                    or samples_since_peak > REFRACTORY_SAMPLES
                )
            ):
                self.detected.append(self.sample_index)
                self.last_rpeak_index = self.sample_index
                self.rpeak_count += 1

        self.sample_index += 1

    def process(self, adc_samples):
        for sample in adc_samples:
            filtered = self.filter_sample(sample)
            self.detect_rpeak(filtered)

        return np.asarray(self.detected, dtype=np.int64)


def greedy_match(reference, detected, tolerance_samples):
    reference = np.asarray(reference, dtype=np.int64)
    detected = np.asarray(detected, dtype=np.int64)

    used = np.zeros(len(detected), dtype=bool)
    matched_errors = []

    tp = 0

    for r in reference:
        candidates = np.where(
            (~used) &
            (np.abs(detected - r) <= tolerance_samples)
        )[0]

        if len(candidates) == 0:
            continue

        best = candidates[np.argmin(np.abs(detected[candidates] - r))]
        used[best] = True
        tp += 1
        matched_errors.append(int(detected[best] - r))

    fn = len(reference) - tp
    fp = int((~used).sum())

    return tp, fp, fn, np.asarray(matched_errors, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default="100")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=100.0,
        help="Matching tolerance around reference beat annotation."
    )
    args = parser.parse_args()

    print(f"Downloading MIT-BIH record {args.record} from PhysioNet...")

    record = wfdb.rdrecord(args.record, pn_dir="mitdb")
    ann = wfdb.rdann(args.record, "atr", pn_dir="mitdb")

    source_fs = int(round(record.fs))
    n_source = min(
        len(record.p_signal),
        int(args.seconds * source_fs)
    )

    signal = record.p_signal[:n_source, 0].astype(np.float64)

    # Remove gross DC before resampling.
    signal = signal - np.median(signal)

    # MIT-BIH is 360 Hz. 25/36 converts exactly to 250 Hz.
    signal_250 = resample_poly(signal, 25, 36)

    # Convert the physical signal to a plausible 12-bit ADC-like stream.
    adc = np.clip(
        np.rint(ADC_MIDPOINT + ADC_COUNTS_PER_MV * signal_250),
        0,
        4095
    ).astype(np.uint16)

    max_ann_source = n_source
    ref = [
        int(round(sample * TARGET_FS / source_fs))
        for sample, symbol in zip(ann.sample, ann.symbol)
        if sample < max_ann_source and symbol in BEAT_SYMBOLS
    ]

    # Ignore first two seconds because firmware deliberately has warm-up.
    warmup = 2 * TARGET_FS
    ref = np.asarray([x for x in ref if x > warmup], dtype=np.int64)

    detector = VitaTrackDetector(TARGET_FS)
    detected = detector.process(adc)
    detected = detected[detected > warmup]

    tolerance_samples = int(
        round(args.tolerance_ms * TARGET_FS / 1000.0)
    )

    tp, fp, fn, errors = greedy_match(
        ref,
        detected,
        tolerance_samples
    )

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )

    print()
    print("=== VitaTrack R-peak validation ===")
    print(f"Record:                 {args.record}")
    print(f"Duration tested:        {args.seconds:.1f} s")
    print(f"Source sampling rate:   {source_fs} Hz")
    print(f"Replay sampling rate:   {TARGET_FS} Hz")
    print(f"Reference beats:        {len(ref)}")
    print(f"Detected beats:         {len(detected)}")
    print(f"True positives:         {tp}")
    print(f"False positives:        {fp}")
    print(f"False negatives:        {fn}")
    print(f"Sensitivity / recall:   {100*sensitivity:.2f}%")
    print(f"Precision:              {100*precision:.2f}%")
    print(f"F1:                     {100*f1:.2f}%")

    if len(errors):
        mae_ms = (
            np.mean(np.abs(errors)) *
            1000.0 / TARGET_FS
        )
        print(f"Mean abs timing error:  {mae_ms:.2f} ms")

    print()
    print("First reference beats:", ref[:10].tolist())
    print("First detected beats: ", detected[:10].tolist())


if __name__ == "__main__":
    main()
