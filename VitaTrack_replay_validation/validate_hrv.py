import argparse
import numpy as np
import wfdb
from scipy.signal import resample_poly

TARGET_FS = 250
ADC_MIDPOINT = 2048.0
ADC_COUNTS_PER_MV = 400.0

REFRACTORY_SAMPLES = 63
HRV_WINDOW_MS = 60000.0
RR_MIN_MS = 300.0
RR_MAX_MS = 2000.0

BEAT_SYMBOLS = {
    "N", "L", "R", "B",
    "A", "a", "J", "S",
    "V", "r", "F",
    "e", "j", "n", "E",
    "/", "f", "Q", "?"
}


class FirmwareDetector:
    """
    Python mirror of the current VitaTrack C filtering + R-peak logic.
    """

    def __init__(self):
        self.ecg_x_prev = 0.0
        self.ecg_hp_prev = 0.0
        self.ecg_lp_prev = 0.0

        self.ecg_filtered_prev = 0.0
        self.ecg_energy_floor = 1.0
        self.ecg_energy_peak = 1.0

        self.ecg_sample_index = 0
        self.last_rpeak_index = 0
        self.rpeak_count = 0

        self.rr = []
        self.detected = []

    def filter_sample(self, raw_sample):
        x = float(raw_sample) - ADC_MIDPOINT

        hp_alpha = 0.9876
        hp = hp_alpha * (
            self.ecg_hp_prev +
            x -
            self.ecg_x_prev
        )

        self.ecg_x_prev = x
        self.ecg_hp_prev = hp

        lp_alpha = 0.4680
        self.ecg_lp_prev = (
            self.ecg_lp_prev +
            lp_alpha * (hp - self.ecg_lp_prev)
        )

        return self.ecg_lp_prev

    def detect_rpeak(self, filtered_sample):
        derivative = (
            filtered_sample -
            self.ecg_filtered_prev
        )

        self.ecg_filtered_prev = filtered_sample
        energy = derivative * derivative

        self.ecg_energy_floor = (
            0.995 * self.ecg_energy_floor +
            0.005 * energy
        )

        if energy > self.ecg_energy_peak:
            self.ecg_energy_peak = energy
        else:
            self.ecg_energy_peak = (
                0.999 * self.ecg_energy_peak +
                0.001 * self.ecg_energy_floor
            )

        threshold = (
            self.ecg_energy_floor +
            0.30 * (
                self.ecg_energy_peak -
                self.ecg_energy_floor
            )
        )

        if self.ecg_sample_index > 2 * TARGET_FS:
            samples_since_peak = (
                self.ecg_sample_index -
                self.last_rpeak_index
            )

            if (
                energy > threshold and
                (
                    self.rpeak_count == 0 or
                    samples_since_peak > REFRACTORY_SAMPLES
                )
            ):
                self.detected.append(self.ecg_sample_index)

                if self.rpeak_count > 0:
                    rr_ms = (
                        1000.0 *
                        samples_since_peak /
                        TARGET_FS
                    )

                    if RR_MIN_MS <= rr_ms <= RR_MAX_MS:
                        self.rr.append(rr_ms)

                self.last_rpeak_index = self.ecg_sample_index
                self.rpeak_count += 1

        self.ecg_sample_index += 1

    def process(self, adc):
        for raw in adc:
            filtered = self.filter_sample(raw)
            self.detect_rpeak(filtered)


def firmware_style_window(rr):
    """
    Mirrors the C selection logic:
    walk backward from newest RR values until accumulated time >= 60 s.
    """
    accumulated = 0.0
    selected = []

    for value in reversed(rr):
        if accumulated >= HRV_WINDOW_MS:
            break
        selected.append(float(value))
        accumulated += float(value)

    selected.reverse()

    if accumulated < HRV_WINDOW_MS or len(selected) < 8:
        return None

    return np.asarray(selected, dtype=np.float64)


def hrv_features(rr):
    rr = np.asarray(rr, dtype=np.float64)

    mean_rr = np.mean(rr)
    d = np.diff(rr)

    return {
        "mean_rr_ms": mean_rr,
        "mean_hr_bpm": 60000.0 / mean_rr,
        "sdnn_ms": np.std(rr, ddof=1),
        "rmssd_ms": np.sqrt(np.mean(d ** 2)),
        "pnn50_pct": 100.0 * np.mean(np.abs(d) > 50.0),
        "rr_range_ms": np.max(rr) - np.min(rr),
        "interval_count": len(rr),
        "window_ms": np.sum(rr),
    }


def reference_rr_from_annotations(
    ann,
    source_fs,
    start_source,
    stop_source
):
    rel_beats_250 = []

    for sample, symbol in zip(ann.sample, ann.symbol):
        if (
            start_source <= sample < stop_source
            and symbol in BEAT_SYMBOLS
        ):
            rel_source = sample - start_source
            rel_250 = int(round(
                rel_source *
                TARGET_FS /
                source_fs
            ))

            # Match firmware warm-up.
            if rel_250 > 2 * TARGET_FS:
                rel_beats_250.append(rel_250)

    rr = []

    for a, b in zip(
        rel_beats_250[:-1],
        rel_beats_250[1:]
    ):
        value = (
            1000.0 *
            (b - a) /
            TARGET_FS
        )

        if RR_MIN_MS <= value <= RR_MAX_MS:
            rr.append(value)

    return np.asarray(rel_beats_250), np.asarray(rr)


def print_features(title, feat):
    print(title)
    print(f"  Interval count : {feat['interval_count']}")
    print(f"  Window duration: {feat['window_ms']:.1f} ms")
    print(f"  Mean RR        : {feat['mean_rr_ms']:.3f} ms")
    print(f"  Mean HR        : {feat['mean_hr_bpm']:.3f} bpm")
    print(f"  SDNN           : {feat['sdnn_ms']:.3f} ms")
    print(f"  RMSSD          : {feat['rmssd_ms']:.3f} ms")
    print(f"  pNN50          : {feat['pnn50_pct']:.3f} %")
    print(f"  RR range       : {feat['rr_range_ms']:.3f} ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", default="100")
    p.add_argument("--start", type=float, default=5.0)
    p.add_argument("--seconds", type=float, default=90.0)
    args = p.parse_args()

    print(
        f"Downloading MIT-BIH record {args.record} "
        f"from PhysioNet..."
    )

    record = wfdb.rdrecord(
        args.record,
        pn_dir="mitdb"
    )

    ann = wfdb.rdann(
        args.record,
        "atr",
        pn_dir="mitdb"
    )

    source_fs = int(round(record.fs))

    start_source = int(
        round(args.start * source_fs)
    )

    stop_source = int(
        round(
            (args.start + args.seconds) *
            source_fs
        )
    )

    signal = record.p_signal[
        start_source:stop_source,
        0
    ].astype(np.float64)

    signal = signal - np.median(signal)

    # 360 Hz -> 250 Hz for MIT-BIH.
    signal_250 = resample_poly(
        signal,
        25,
        36
    )

    adc = np.clip(
        np.rint(
            ADC_MIDPOINT +
            ADC_COUNTS_PER_MV * signal_250
        ),
        0,
        4095
    ).astype(np.uint16)

    fw = FirmwareDetector()
    fw.process(adc)

    fw_window = firmware_style_window(fw.rr)

    if fw_window is None:
        raise RuntimeError(
            "Firmware-equivalent detector did not "
            "accumulate a full 60-second HRV window."
        )

    ref_beats, ref_rr = (
        reference_rr_from_annotations(
            ann,
            source_fs,
            start_source,
            stop_source
        )
    )

    ref_window = firmware_style_window(ref_rr)

    if ref_window is None:
        raise RuntimeError(
            "Reference annotations did not produce "
            "a full 60-second HRV window."
        )

    fw_feat = hrv_features(fw_window)
    ref_feat = hrv_features(ref_window)

    print()
    print("=== VitaTrack HRV validation ===")
    print(f"Record           : {args.record}")
    print(f"Segment          : {args.start:.1f} s -> "
          f"{args.start + args.seconds:.1f} s")
    print(f"Replay samples   : {len(adc)}")
    print(f"Detected R-peaks : {len(fw.detected)}")
    print(f"Reference beats  : {len(ref_beats)}")
    print()

    print_features(
        "Firmware-equivalent HRV:",
        fw_feat
    )

    print()

    print_features(
        "Annotation-reference HRV:",
        ref_feat
    )

    print()
    print("Feature differences "
          "(firmware-equivalent - reference):")

    keys = [
        "mean_rr_ms",
        "mean_hr_bpm",
        "sdnn_ms",
        "rmssd_ms",
        "pnn50_pct",
        "rr_range_ms",
    ]

    for key in keys:
        diff = fw_feat[key] - ref_feat[key]
        denom = abs(ref_feat[key])

        pct = (
            100.0 * diff / denom
            if denom > 1e-12
            else 0.0
        )

        print(
            f"  {key:14s}: "
            f"{diff:+.6f} "
            f"({pct:+.4f}%)"
        )

    print()
    print("Firmware debugger target values:")
    print(
        f"  hrv_features.mean_rr_ms   = "
        f"{fw_feat['mean_rr_ms']:.6f}f"
    )
    print(
        f"  hrv_features.mean_hr_bpm  = "
        f"{fw_feat['mean_hr_bpm']:.6f}f"
    )
    print(
        f"  hrv_features.sdnn_ms      = "
        f"{fw_feat['sdnn_ms']:.6f}f"
    )
    print(
        f"  hrv_features.rmssd_ms     = "
        f"{fw_feat['rmssd_ms']:.6f}f"
    )
    print(
        f"  hrv_features.pnn50_pct    = "
        f"{fw_feat['pnn50_pct']:.6f}f"
    )
    print(
        f"  hrv_features.rr_range_ms  = "
        f"{fw_feat['rr_range_ms']:.6f}f"
    )
    print(
        f"  hrv_features.interval_count = "
        f"{fw_feat['interval_count']}"
    )


if __name__ == "__main__":
    main()
