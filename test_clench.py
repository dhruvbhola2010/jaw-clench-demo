import argparse
import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, FilterTypes

parser = argparse.ArgumentParser()
parser.add_argument("serial_port", help="Serial port for the NeuroPawn Knight board")
args = parser.parse_args()

BoardShim.set_log_level(LogLevels.LEVEL_OFF.value)

params = BrainFlowInputParams()
params.serial_port = args.serial_port

board = BoardShim(BoardIds.NEUROPAWN_KNIGHT_BOARD.value, params)
board.prepare_session()
board.start_stream(450000)
print("Stream started, activating channels...")
time.sleep(2)

for ch in range(1, 3):
    time.sleep(0.5)
    board.config_board(f"chon_{ch}_12")
    time.sleep(1)
    board.config_board(f"rldadd_{ch}")
    time.sleep(0.5)

print("Settling...")
time.sleep(3)

sampling_rate = BoardShim.get_sampling_rate(board.get_board_id())
eeg_channels = BoardShim.get_exg_channels(board.get_board_id())
window = int(sampling_rate * 0.3)

print(f"Sampling rate: {sampling_rate}Hz")
print(f"\nSit still for 5s, then CLENCH for 5s, then relax.")
print("Watch the numbers:\n")

try:
    while True:
        time.sleep(0.2)
        data = board.get_current_board_data(window)
        if data.shape[1] < 5:
            continue

        powers = []
        for ch_idx in eeg_channels[:2]:
            filtered = data[ch_idx].copy()
            DataFilter.perform_bandpass(filtered, sampling_rate, 20.0, 50.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            powers.append(np.sqrt(np.mean(filtered ** 2)))

        bar = "#" * min(int(max(powers) / 10), 80)
        print(f"  ch1={powers[0]:8.1f}  ch2={powers[1]:8.1f}  |{bar}", flush=True)

except KeyboardInterrupt:
    pass

board.stop_stream()
board.release_session()
print("\nDone.")
