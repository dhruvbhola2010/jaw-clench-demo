import argparse
import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels

parser = argparse.ArgumentParser()
parser.add_argument("serial_port", help="Serial port for the NeuroPawn Knight board")
args = parser.parse_args()

BoardShim.set_log_level(LogLevels.LEVEL_OFF.value)

params = BrainFlowInputParams()
params.serial_port = args.serial_port

board = BoardShim(BoardIds.NEUROPAWN_KNIGHT_BOARD.value, params)
board.prepare_session()
board.start_stream(450000)
print("Stream started")
time.sleep(2)

for ch in range(1, 3):
    time.sleep(0.5)
    board.config_board(f"chon_{ch}_12")
    time.sleep(1)
    board.config_board(f"rldadd_{ch}")
    time.sleep(0.5)

print("Channels on. Waiting 3s to settle...")
time.sleep(3)

sampling_rate = BoardShim.get_sampling_rate(board.get_board_id())
eeg_channels = BoardShim.get_exg_channels(board.get_board_id())
print(f"Sampling rate: {sampling_rate}, EEG channels: {eeg_channels}")

print("\n--- Testing get_board_data() ---")
data = board.get_board_data()
print(f"  shape: {data.shape}")
if data.shape[1] > 0:
    for ch in eeg_channels[:2]:
        print(f"  ch{ch} range: [{data[ch].min():.1f}, {data[ch].max():.1f}]")

print("\n--- Testing get_current_board_data(50) every 0.5s for 5s ---")
for i in range(10):
    time.sleep(0.5)
    data = board.get_current_board_data(50)
    if data.shape[1] > 0:
        vals = [f"ch{ch}=[{data[ch].min():.0f},{data[ch].max():.0f}]" for ch in eeg_channels[:2]]
        print(f"  {i}: shape={data.shape}, {', '.join(vals)}")
    else:
        print(f"  {i}: shape={data.shape} (empty)")

board.stop_stream()
board.release_session()
print("\nDone.")
