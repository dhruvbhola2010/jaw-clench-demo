# Jaw Clench Yes/No Demo

An early Python prototype that uses jaw-muscle activity (EMG) recorded with a NeuroPawn Knight board. A single clench produces “no” and two clenches within one second produce “yes.” The demo prints the result and speaks it using the macOS `say` command.

The detector applies a 20–50 Hz band-pass filter, computes RMS amplitude over a rolling window, and compares it against a threshold derived from a short rest calibration and a fixed minimum.

## Run locally

Use Python 3, a NeuroPawn Knight board, and a macOS terminal. Connect the board and replace the example serial port with the port on your machine. On macOS, `ls /dev/cu.usb*` can help locate it.

```sh
python3 -m pip install -r requirements.txt
python3 jaw_clench_demo.py /dev/cu.usbserial-EXAMPLE
```

Follow the prompts to record a resting baseline and try a few clenches. Press Ctrl+C to stop.

## Diagnostic scripts

- `test_board.py` checks whether the board is streaming samples.
- `test_clench.py` prints filtered signal amplitude during rest and clenching.

Both scripts take the same serial-port argument as the demo.

## Prototype limitations

The threshold settings were developed for one hardware setup and may need adjustment. A sustained clench can register twice because the current detector does not require a release between detections.
