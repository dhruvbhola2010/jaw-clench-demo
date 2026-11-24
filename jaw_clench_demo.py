# time     — lets us do time.sleep() to pause, and time.time() to get current time
# sys      — gives us access to sys.stdin which is keyboard input
# select   — lets us check if a key was pressed WITHOUT waiting/blocking
# subprocess — lets us run terminal commands (we use it to run "say" for audio)
# numpy    — math library. we use it for squaring, averaging, square root
import argparse
import time
import sys
import select
import subprocess
import numpy as np

# brainflow is the library that talks to the NeuroPawn board over USB
# BoardShim      — the main class that controls the board (start, stop, read data)
# BrainFlowInputParams — a container where we put settings like which USB port to use
# BoardIds       — a list of all supported boards. ours is NEUROPAWN_KNIGHT_BOARD
# LogLevels      — controls how much debug info brainflow prints
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels

# DataFilter  — has all the signal processing functions (filters)
# FilterTypes — types of filters available. we use BUTTERWORTH (a smooth filter)
# NoiseTypes  — types of electrical noise. SIXTY = 60Hz (US power grid frequency)
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes

# tty + termios — these let us change how the terminal handles keyboard input
# normally you type something and press Enter. we change it so any keypress
# is detected instantly without needing Enter. we restore it when we exit.
import tty
import termios

# === SETTINGS ===

# after you do the first clench, you have this many seconds to do a second
# clench for it to count as a "double clench" (which means "yes")
DOUBLE_CLENCH_WINDOW = 1.0

# after the computer says "yes" or "no", it ignores all signals for this
# many seconds. prevents it from firing again from the same clench.
COOLDOWN = 2.0

# the minimum power level that counts as a jaw clench.
# we set this to 15000 because your board's resting noise goes up to ~13000
# so anything below 15000 is definitely not a real clench.
MIN_THRESHOLD = 15000

# we need the signal to be above the threshold for this many checks in a row
# before we believe it's a real clench. we check 10 times per second, so
# 3 checks = the signal was strong for at least 0.3 seconds.
# a real jaw clench lasts 0.3+ seconds. a noise spike lasts 1-2 checks max.
# this is how we tell real clenches apart from random noise.
CONFIRM_READINGS = 3


def key_pressed():
    # select.select() checks: "is there keyboard input available right now?"
    # the three lists are: [read from], [write to], [errors] — we only care about reading
    # the 0 at the end means "don't wait at all, just check instantly"
    # if there IS a key waiting, read it with sys.stdin.read(1) (1 = one character)
    # if not, return None (meaning "no key was pressed")
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def say(word):
    # "say" is a built-in macOS terminal command that speaks text out loud
    # subprocess.Popen starts the command in the background
    # so the program doesn't pause while the word is being spoken
    # if we used subprocess.run() instead, the program would freeze
    # until the word finished playing
    subprocess.Popen(["say", word])


def setup_board(serial_port):
    # turn off brainflow's debug logging
    # without this, it prints hundreds of "Wrong end byte" warnings per second
    # those warnings are just serial communication hiccups — harmless but noisy
    BoardShim.set_log_level(LogLevels.LEVEL_OFF.value)

    # create a params object and tell it which USB port the board is on
    params = BrainFlowInputParams()
    params.serial_port = serial_port

    # create the board object
    # BoardIds.NEUROPAWN_KNIGHT_BOARD.value is just a number (like 63 or whatever)
    # that tells brainflow "this is a NeuroPawn Knight Board on the other end"
    # so it knows what communication protocol to use
    board = BoardShim(BoardIds.NEUROPAWN_KNIGHT_BOARD.value, params)

    # prepare_session() opens the USB connection to the board
    board.prepare_session()

    # start_stream() tells the board "start sending me EEG data"
    # 450000 is the ring buffer size — it can hold 450,000 samples
    # at 125 samples/second, that's about 1 hour of data
    # the board continuously writes new data into this buffer
    # and we read from it whenever we want
    board.start_stream(450000)
    print("Stream started, activating channels...")

    # wait 2 seconds for the board to get going
    time.sleep(2)

    # "chon_1_12" is a command string the board understands:
    #   "chon" = channel on
    #   "1"    = channel number 1 (the first electrode)
    #   "12"   = gain of 12x (how much to amplify the tiny brain signal)
    # we only use channel 1 because testing showed channel 2 always reads zero
    board.config_board("chon_1_12")
    print("  Channel 1 on")
    time.sleep(1)

    # "rldadd_1" = add channel 1 to the Right Leg Drive circuit
    # the ear clip electrode is the reference point (RLD/COMM)
    # this command connects channel 1's signal to be compared against the ear clip
    # without this, the board can't subtract out common noise and you'd just see garbage
    board.config_board("rldadd_1")
    print("  RLD added for channel 1")
    time.sleep(0.5)

    # wait 3 more seconds for the electrical signal to stabilize
    # right after turning on, the signal is jumpy as the electronics settle
    print("Letting signal settle...")
    time.sleep(3)

    return board


def get_emg_power(data, ch_idx, sampling_rate):
    # data is a 2D array (matrix). each row is a channel, each column is a time point.
    # data looks like:
    #   row 0: timestamps
    #   row 1: channel 1 values    <-- this is what we want
    #   row 2: channel 2 values
    #   row 3: channel 3 values
    #   ... etc
    # ch_idx tells us which row to grab (for channel 1, ch_idx = 1)
    channel_data = data[ch_idx]

    # if we have fewer than 10 data points, not enough to filter — return 0
    if len(channel_data) < 10:
        return 0

    # make a copy because the filter functions modify the array in place
    # (they change the original data, which we don't want)
    filtered = channel_data.copy()

    # FILTER 1: remove 60Hz electrical noise
    # every wall outlet in the US puts out 60Hz electromagnetic waves
    # your scalp electrode is basically an antenna that picks this up
    # this is NOT brain activity — it's just electrical pollution
    # this filter specifically targets and removes the 60Hz frequency
    DataFilter.remove_environmental_noise(filtered, sampling_rate, NoiseTypes.SIXTY.value)

    # FILTER 2: bandpass filter — only keep frequencies between 20Hz and 50Hz
    # "bandpass" means "let this band pass through, block everything else"
    #
    # why 20-50Hz? because that's where jaw muscle signals (EMG) live.
    # brain waves are mostly below 30Hz.
    # jaw muscles fire rapidly at 20-50Hz and produce a MUCH stronger signal.
    # so by only looking at 20-50Hz, we mostly see jaw clenches and not brain waves.
    #
    # the other arguments:
    #   4 = filter "order" (higher = sharper cutoff, 4 is standard)
    #   BUTTERWORTH = a type of filter that has a smooth response (no ripple)
    #   0 = ripple value (not used for butterworth, so just 0)
    DataFilter.perform_bandpass(filtered, sampling_rate, 20.0, 50.0, 4,
                                FilterTypes.BUTTERWORTH.value, 0)

    # now 'filtered' contains only the 20-50Hz jaw muscle signal, with 60Hz noise removed
    #
    # we need to turn this wave into a single number that says "how strong is it?"
    # that's what RMS (Root Mean Square) does:
    #   1. square every value        — makes all values positive (no negatives)
    #   2. take the mean (average)   — get the average squared value
    #   3. take the square root      — undo the squaring to get back to original units
    #
    # result:
    #   resting (no clench):  ~500 to ~2000
    #   clenching jaw:        ~20000 to ~50000
    return np.sqrt(np.mean(filtered ** 2))


def calibrate(board, ch_idx, sampling_rate):
    print(f"\n--- CALIBRATION ---")
    print(f"Relax your jaw for 5 seconds. Press 1 to skip.")
    time.sleep(2)
    print("Recording baseline...")

    # window = how many samples to grab each time we read from the board
    # sampling_rate is 125 (125 samples per second)
    # 0.5 * 125 = 62 samples = half a second of data
    window = int(sampling_rate * 0.5)

    # we'll collect power readings while you sit still
    # these tell us what your "resting" signal looks like
    powers = []
    start = time.time()

    # keep reading for 5 seconds
    while time.time() - start < 5:
        time.sleep(0.3)  # check every 0.3 seconds

        # if user pressed "1" and we have at least 3 readings, skip ahead
        if key_pressed() == "1" and len(powers) >= 3:
            break

        # get_current_board_data(window) peeks at the last 'window' samples
        # in the ring buffer. it does NOT remove them — it's just a peek.
        # this means if we call it twice quickly, we might see the same data.
        # data.shape[1] is the number of columns (time points) we actually got.
        data = board.get_current_board_data(window)
        if data.shape[1] < 10:  # not enough data yet
            continue

        # run the signal processing and get the power number
        p = get_emg_power(data, ch_idx, sampling_rate)
        powers.append(p)
        print(f"  resting: {p:.0f}")

    # if we got zero readings, something is wrong
    if not powers:
        print("  ERROR: No data from board!")
        board.stop_stream()
        board.release_session()
        exit(1)

    # find the highest power we saw while you were resting
    baseline_max = np.max(powers)

    # set the threshold — signal must be above this to count as a clench
    # we pick whichever is HIGHER:
    #   baseline_max * 2  — twice your worst resting noise (adapts to your signal)
    #   MIN_THRESHOLD     — hard minimum of 15000 (safety net because your noise
    #                       sometimes spikes to ~13000 even when you're not clenching)
    threshold = max(baseline_max * 2, MIN_THRESHOLD)

    print(f"\n  Baseline max: {baseline_max:.0f}")
    print(f"  Clench threshold: {threshold:.0f}")

    # PHASE 2: test that clenches actually work
    print(f"\nClench your jaw hard 3 times. Press 1 to skip.")
    time.sleep(1)

    clenches = 0  # how many clenches we detected
    last = 0      # timestamp of the last detected clench (for debouncing)
    start = time.time()

    # give the user 10 seconds to do test clenches
    while time.time() - start < 10:
        if key_pressed() == "1":
            break
        time.sleep(0.15)

        data = board.get_current_board_data(window)
        if data.shape[1] < 10:
            continue

        p = get_emg_power(data, ch_idx, sampling_rate)
        now = time.time()

        # if power is above threshold AND it's been at least 0.8 seconds
        # since the last clench (so one long clench doesn't count as 5 clenches)
        if p > threshold and now - last > 0.8:
            clenches += 1
            last = now
            print(f"  Clench #{clenches}! power={p:.0f}")

    if clenches:
        print(f"\n  {clenches} clenches detected — good to go!")
    else:
        print(f"\n  No clenches detected — try clenching harder.")

    # return the threshold so the main loop can use it
    return threshold


def main():
    parser = argparse.ArgumentParser(description="Demo single- and double-clench inputs.")
    parser.add_argument("serial_port", help="Serial port for the NeuroPawn Knight board")
    args = parser.parse_args()

    # save the current terminal settings so we can restore them later
    # (when the program exits, we put the terminal back to normal)
    old_settings = termios.tcgetattr(sys.stdin)

    # setcbreak makes every keypress available immediately
    # normally the terminal waits until you press Enter
    # we need instant keypresses so pressing "1" skips calibration right away
    tty.setcbreak(sys.stdin.fileno())

    # connect to the board and set it up
    board = setup_board(args.serial_port)

    # ask brainflow what board ID we're using
    board_id = board.get_board_id()

    # get_exg_channels returns a list like [1, 2, 3, 4, 5, 6, 7, 8]
    # these are the row indices in the data matrix for EEG/EMG channels
    eeg_channels = BoardShim.get_exg_channels(board_id)

    # we only use the first channel (index 1) because channel 2 reads all zeros
    ch_idx = eeg_channels[0]

    # get_sampling_rate returns 125 (the board takes 125 readings per second)
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    # window = how many samples to read each loop iteration
    # 0.3 * 125 = 37 samples = 0.3 seconds of data
    window = int(sampling_rate * 0.3)
    print(f"Sampling rate: {sampling_rate}Hz, using channel {ch_idx}")

    # run calibration — this returns the threshold number
    threshold = calibrate(board, ch_idx, sampling_rate)

    print("\n--- RUNNING ---")
    print("  Single clench  → 'no'")
    print("  Double clench  → 'yes'")
    print("  Ctrl+C to quit\n")

    # === STATE VARIABLES ===
    # these track what's happening over time

    # when did we last say "yes" or "no"? (used for cooldown)
    # 0 = we haven't said anything yet
    last_action_time = 0

    # when did the first clench happen? (used to detect single vs double)
    # None = no clench detected yet, waiting for one
    first_clench_time = None

    # when was ANY clench last detected? (used for debouncing)
    # prevents the same clench from being counted twice
    last_clench_time = 0

    # how many checks in a row has the signal been above the threshold?
    # needs to hit 3 before we believe it's a real clench
    above_count = 0

    # try/except/finally:
    #   try     — run the main loop
    #   except  — if user presses Ctrl+C, catch it gracefully
    #   finally — no matter what happens, always clean up (restore terminal, stop board)
    try:
        # the main loop — runs forever until Ctrl+C
        while True:
            # pause 0.1 seconds between each check (10 checks per second)
            time.sleep(0.1)

            # get the current time (seconds since 1970, like 1753283200.5)
            now = time.time()

            # COOLDOWN CHECK:
            # if we said "yes" or "no" less than 2 seconds ago, skip this check
            # this prevents the same clench from triggering multiple times
            if now - last_action_time < COOLDOWN:
                above_count = 0  # reset the counter during cooldown
                continue         # skip to the next iteration of the while loop

            # read the last 37 samples (0.3 seconds) from the board's buffer
            # this is a "peek" — it doesn't remove the data from the buffer
            data = board.get_current_board_data(window)

            # if there aren't at least 10 data points, skip (not enough to process)
            if data.shape[1] < 10:
                continue

            # run the signal processing pipeline:
            #   raw data → remove 60Hz noise → bandpass 20-50Hz → RMS → one number
            power = get_emg_power(data, ch_idx, sampling_rate)

            # is the power above our threshold?
            if power > threshold:
                above_count += 1   # yes — one more consecutive strong reading
            else:
                above_count = 0    # no — reset the counter to zero
                                   # this is key: a noise spike might be strong for
                                   # 1 or 2 reads, but then it drops. a real clench
                                   # stays strong for 3+ reads. by resetting to 0
                                   # whenever it drops, we filter out noise.

            # CLENCH CONFIRMED:
            # above_count >= 3 means the signal was strong for 3 reads in a row (0.3s)
            # now - last_clench_time > 0.5 means at least 0.5s since last detection
            if above_count >= CONFIRM_READINGS and (now - last_clench_time > 0.5):
                above_count = 0        # reset counter
                last_clench_time = now  # record when this clench happened

                # is this the FIRST clench or the SECOND?
                if first_clench_time is None:
                    # FIRST CLENCH — no clench was pending
                    # save the time and wait up to 1 second for a possible second clench
                    first_clench_time = now
                    print(f"  Clench! (power={power:.0f}) waiting for double...", flush=True)

                elif now - first_clench_time < DOUBLE_CLENCH_WINDOW:
                    # SECOND CLENCH — and it happened within 1 second of the first
                    # that's a double clench = "YES"
                    print("→ YES (double clench)", flush=True)
                    say("yes")               # speak "yes" out loud
                    first_clench_time = None  # reset — no clench pending anymore
                    last_action_time = now    # start the 2-second cooldown

            # TIMEOUT CHECK:
            # if there IS a first clench pending, and 1 second has passed
            # without a second clench, that's a single clench = "NO"
            elif first_clench_time is not None and now - first_clench_time >= DOUBLE_CLENCH_WINDOW:
                print("→ NO (single clench)", flush=True)
                say("no")                    # speak "no" out loud
                first_clench_time = None     # reset — no clench pending anymore
                last_action_time = now       # start the 2-second cooldown

    except KeyboardInterrupt:
        # user pressed Ctrl+C — this is how you stop the program
        print("\nStopping...")

    finally:
        # this ALWAYS runs, even if there was an error or Ctrl+C
        # restore the terminal to normal (undo the setcbreak from earlier)
        # so your terminal doesn't stay in weird raw-input mode
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # tell the board to stop sending data
        board.stop_stream()

        # close the USB connection
        board.release_session()
        print("Done.")


# this is a python convention:
# if this file is run directly (python3 blink_detector.py), run main()
# if this file is imported by another file, don't run anything
if __name__ == "__main__":
    main()
