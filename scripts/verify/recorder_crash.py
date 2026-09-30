"""Kill the application mid-recording, start it again, and see what survived.

    python scripts\verify\recorder_crash.py FOLDER [--record-s S]

A crash is the one moment a recorder cannot tidy up after itself: no
header is finished, no sidecar is closed. This makes one on purpose --
TerminateProcess, the same abrupt end a crash or a power button gives the
process -- and then checks the three things that matter:

  1. just after the kill the segment really is unfinished (state "open"),
     so the test is testing something
  2. the next start repairs it: readable, state "recovered"
  3. how much was lost: the audio on disk against the time the process was
     recording before it was killed. The bound is FLUSH_S (0.5 s) of
     frames still inside the process, plus what was queued.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scipy.io import wavfile

from friture.recording.wav_segment import SegmentInfo, read_header

args = sys.argv[1:]
folder = Path(args[0])
# Not a multiple of HEADER_REFRESH_S (5 s): killed at 40.0 s the header had
# just been rewritten and happened to be exact, which says nothing either way.
record_s = float(args[args.index("--record-s") + 1]) if "--record-s" in args else 42.3
runner = [sys.executable, "-u", str(ROOT / "scripts" / "verify" / "recorder_long_run.py"), str(folder)]
ok = True


def check(label, cond, detail=""):
    global ok
    print("%-56s %s   %s" % (label, "ok " if cond else "FAIL", detail))
    ok = ok and cond


folder.mkdir(parents=True, exist_ok=True)
before = {p.name for p in folder.glob("*.wav")}

print("recording for %.0f s, then killing the process outright ..." % record_s)
proc = subprocess.Popen(runner + ["--minutes", "100", "--stall-at", "999", "--stop-at", "999"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
deadline = time.time() + 60
while time.time() < deadline and not ({p.name for p in folder.glob("*.wav")} - before):
    time.sleep(0.2)
new = sorted({p.name for p in folder.glob("*.wav")} - before)
if not new:
    proc.kill()
    print("FAIL: nothing was recorded within 60 s")
    sys.exit(1)
time.sleep(record_s)
kill_wall = time.time()
proc.kill()                                    # TerminateProcess: no cleanup at all
proc.wait()

victim = sorted({p.name for p in folder.glob("*.wav")} - before)[-1]
side = SegmentInfo.load(folder / (Path(victim).stem + ".json"))
_, channels, claimed = read_header(folder / victim)
size_frames = ((folder / victim).stat().st_size - 44) // (2 * channels)
check("the killed segment was left unfinished", side.state == "open", "state %s" % side.state)
# Informational: whether the header is behind depends on where between two
# refreshes the kill lands. Repair must work either way, so it is not a check.
print("%-56s %s   %s" % ("(its header at the moment of the kill)", "info",
                           "header %d frames, file %d frames"
                           % (claimed // (2 * channels), size_frames)))

print("\nstarting the application again, briefly, so the recorder repairs it ...")
subprocess.run(runner + ["--minutes", "0.3", "--stall-at", "999", "--stop-at", "999"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)

side = SegmentInfo.load(folder / (Path(victim).stem + ".json"))
check("repaired on the next start", side.state == "recovered", "state %s" % side.state)
try:
    fs, data = wavfile.read(folder / victim, mmap=True)
    frames = data.shape[0]
    del data
    check("readable, and the header now agrees with the file",
          frames == side.n_frames, "%d frames = %.3f s" % (frames, frames / fs))
except Exception as e:
    check("readable, and the header now agrees with the file", False, str(e))
    frames, fs = 0, side.fs
expected = kill_wall - side.start_epoch
lost = expected - frames / fs
check("lost at most FLUSH_S plus a little queue", 0.0 <= lost < 1.0,
      "recording ran %.3f s before the kill, %.3f s on disk, %.3f s lost" % (expected, frames / fs, lost))

print("\n%s" % ("ALL OK" if ok else "FAILURES"))
sys.exit(0 if ok else 1)
