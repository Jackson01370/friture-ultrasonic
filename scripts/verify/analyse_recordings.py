"""Write the analysis log for recordings that do not have one -- or all of them again.

    python scripts\verify\analyse_recordings.py FOLDER [--out DIR] [--force]

Runs friture.recording.analysis, the same code the recorder runs live, over
the WAV segments in FOLDER, in time order. Segments that continue each other
(same run, the indices meet) are analysed as one stream, so a window that
runs over a file boundary is analysed whole, exactly as it was live; at a
break the windows in progress are cut, as live.

--out writes the logs somewhere else instead of beside the audio -- that is
how a live log is compared with an offline one (recording_check.py
--compare-with). Without --force, segments that already have a log are
skipped, and the stream is restarted after each skipped one.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scipy.io import wavfile

from friture.recording.analysis import LiveAnalysis, Where
from friture.recording.analysis_log import AnalysisLog, log_path
from friture.recording.store import SegmentStore

args = sys.argv[1:]
folder = Path(args[0])
out = Path(args[args.index("--out") + 1]) if "--out" in args else folder
force = "--force" in args
out.mkdir(parents=True, exist_ok=True)
CHUNK = 1 << 18

segs = SegmentStore(folder).segments()
if not segs:
    print("no segments in %s" % folder)
    sys.exit(1)

# The logs are APPENDED to -- a window that began in one file is written to
# that file's log after the next file is read -- so a log being redone has
# to go first. Without this, --force added a second set of records after
# the first, and a reader met both.
if force:
    for s in segs:
        old = log_path(out, s.wav[:-4])
        if old.exists():
            old.unlink()
fs = segs[0].fs
analysis = LiveAnalysis(fs)
log = AnalysisLog(fs)
prev = None
done = skipped = 0
t0 = time.perf_counter()
audio_s = 0.0


def emit(records):
    for r in records:
        log.write(out, r)


for s in segs:
    stem = s.wav[:-4]
    if not force and log_path(out, stem).exists():
        emit(analysis.flush())
        prev = None
        skipped += 1
        continue
    contiguous = (prev is not None and s.run_id == prev.run_id
                  and s.run_index == prev.run_index + prev.n_frames)
    if prev is not None and not contiguous:
        emit(analysis.flush())
    rate, data = wavfile.read(folder / s.wav, mmap=True)
    n = min(data.shape[0], s.n_frames)
    for i in range(0, n, CHUNK):
        where = Where(stem, i, s.run_id, s.run_index + i, s.start_epoch + i / fs)
        emit(analysis.feed(data[i:min(i + CHUNK, n)], where))
    del data
    audio_s += n / fs
    done += 1
    prev = s
    print("   %s  %.1f s" % (s.wav, n / fs), flush=True)
emit(analysis.flush())
log.close()
took = time.perf_counter() - t0
print("\nanalysed %d segment(s), %.1f min of audio, in %.1f s (%.0fx real time); skipped %d"
      % (done, audio_s / 60, took, audio_s / max(took, 1e-9), skipped))
