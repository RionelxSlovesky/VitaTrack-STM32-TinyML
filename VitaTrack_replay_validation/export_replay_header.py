import argparse
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

TARGET_FS = 250
ADC_MIDPOINT = 2048.0
ADC_COUNTS_PER_MV = 400.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", default="100")
    p.add_argument("--start", type=float, default=5.0)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--out", default="ecg_replay.h")
    args = p.parse_args()

    record = wfdb.rdrecord(args.record, pn_dir="mitdb")
    source_fs = int(round(record.fs))

    start = int(args.start * source_fs)
    stop = int((args.start + args.seconds) * source_fs)

    signal = record.p_signal[start:stop, 0].astype(np.float64)
    signal = signal - np.median(signal)
    signal_250 = resample_poly(signal, 25, 36)

    adc = np.clip(
        np.rint(ADC_MIDPOINT + ADC_COUNTS_PER_MV * signal_250),
        0,
        4095
    ).astype(np.uint16)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#ifndef ECG_REPLAY_H\n")
        f.write("#define ECG_REPLAY_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define ECG_REPLAY_FS_HZ {TARGET_FS}U\n")
        f.write(f"#define ECG_REPLAY_LENGTH {len(adc)}U\n\n")
        f.write("static const uint16_t ecg_replay_data[ECG_REPLAY_LENGTH] = {\n")

        for i in range(0, len(adc), 16):
            chunk = ", ".join(str(int(x)) for x in adc[i:i+16])
            if i + 16 < len(adc):
                f.write("    " + chunk + ",\n")
            else:
                f.write("    " + chunk + "\n")

        f.write("};\n\n")
        f.write("#endif /* ECG_REPLAY_H */\n")

    print(f"Wrote {out}")
    print(f"Samples: {len(adc)}")
    print(f"Duration: {len(adc)/TARGET_FS:.3f} s")

if __name__ == "__main__":
    main()
