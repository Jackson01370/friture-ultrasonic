"""Judge a folder of continuous recordings: is anything missing, doubled or broken?

    python scripts\verify\recording_check.py FOLDER [--segment-s S]

Checks, and says which one failed:

  READABLE   every WAV opens, and the frame count in its header, in its
             sidecar and implied by its size all agree
  FINISHED   no segment is still "open" (a crash leaves one; the recorder
             repairs it on the next start)
  SEAMLESS   within one capture run the segments join exactly: each starts
             at the frame after the last one ended and names it in
             "continues"; every planned split is exactly S seconds long
  EXPLAINED  every break in the chain carries a gap_reason
  ALIVE      the audio is not silence -- a microphone in a room always
             hears something, so all-zero audio means a broken capture

If recorder_long_run.py left a _run_log.jsonl beside the files, one more:

  COMPLETE   within each capture run, the frames the capture thread read
             kept pace with the audio device clock. A ring buffer that
             overflowed shows up here as seconds missing.

And for the analysis logs (<stem>.analysis.jsonl) -- skip with --no-analysis:

  LOGGED     every segment of a second or more has a log with a header, and
             every record in it points inside that segment
  COVERED    per run, the level records add up to the run's duration (a
             level record is one second) and none of them overlap
  CONTAINED  no record's window runs past the end of its capture run

  --compare-with DIR  compares these logs with the logs in DIR (made by
             analyse_recordings.py --out DIR from the same audio): every
             record must match except for the wall-clock time, which live
             is an estimate per block and offline is computed.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.io import wavfile

from friture.recording.store import SegmentStore
from friture.recording.wav_segment import HEADER_BYTES, read_header

args = sys.argv[1:]
folder = Path(args[0])
segment_s = float(args[args.index("--segment-s") + 1]) if "--segment-s" in args else 600.0

store = SegmentStore(folder)
segs = store.segments()
problems = []


def fail(what, detail):
    problems.append((what, detail))
    print("   FAIL %-10s %s" % (what, detail))


print("%d segments in %s, %.2f GB\n" % (len(segs), folder, store.total_bytes() / 1e9))
print("%-28s %-9s %9s %8s  %-28s %s" % ("segment", "state", "seconds", "drift", "continues", "gap before"))
for s in segs:
    wav = folder / s.wav
    fs, channels, data_bytes = read_header(wav)
    size_frames = (wav.stat().st_size - HEADER_BYTES) // (channels * 2)
    header_frames = data_bytes // (channels * 2)
    try:
        rate, data = wavfile.read(wav, mmap=True)
        read_frames = data.shape[0]
        mid = np.asarray(data[read_frames // 2: read_frames // 2 + min(250_000, read_frames)], dtype=np.float64)
        rms = float(np.sqrt(np.mean(mid ** 2))) if mid.size else 0.0
        del data
    except Exception as e:
        read_frames, rms = -1, 0.0
        fail("READABLE", "%s: %s" % (s.wav, e))
    gap = ("%.3f s (%s)" % (s.gap_before_s, s.gap_reason) if s.gap_before_s is not None
           else (s.gap_reason or "-"))
    print("%-28s %-9s %9.3f %8s  %-28s %s"
          % (s.wav, s.state, s.n_frames / s.fs,
             "%+.1f ppm" % s.clock_drift_ppm if s.clock_drift_ppm is not None else "-",
             s.continues or "-", gap))
    if not (s.n_frames == header_frames == size_frames == read_frames):
        fail("READABLE", "%s: sidecar %d, header %d, size %d, read %d frames"
             % (s.wav, s.n_frames, header_frames, size_frames, read_frames))
    if s.state == "open":
        fail("FINISHED", "%s is still open" % s.wav)
    if rms == 0.0 and s.n_frames > 0:
        fail("ALIVE", "%s is digital silence" % s.wav)

print()
runs = defaultdict(list)
for s in segs:
    runs[s.run_id].append(s)
for run_id, chain in runs.items():
    chain.sort(key=lambda s: s.run_index)
    frames = sum(s.n_frames for s in chain)
    print("run %s: %d segment(s), %.3f s" % (run_id, len(chain), frames / chain[0].fs))
    for prev, nxt in zip(chain, chain[1:]):
        if nxt.run_index != prev.run_index + prev.n_frames:
            if not nxt.gap_reason:
                fail("EXPLAINED", "%s -> %s: %d frames apart and no reason given"
                     % (prev.wav, nxt.wav, nxt.run_index - prev.run_index - prev.n_frames))
        else:
            if nxt.continues != prev.wav:
                fail("SEAMLESS", "%s joins %s exactly but does not say it continues it" % (nxt.wav, prev.wav))
            planned = int(round(segment_s * prev.fs))
            if prev.n_frames != planned:
                fail("SEAMLESS", "%s ends a planned split at %d frames, not %d"
                     % (prev.wav, prev.n_frames, planned))
    for s in chain[1:]:
        if s.gap_before_s is not None or s.gap_reason:
            if s.continues:
                fail("SEAMLESS", "%s claims both a gap and a continuation" % s.wav)

log_path = folder / "_run_log.jsonl"
if log_path.exists():
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_run = defaultdict(list)
    for r in rows:
        if r.get("stream_time") is not None:
            by_run[r["run_id"]].append(r)
    print("\ncapture completeness, from the run log (frames read vs the device clock):")
    fs = segs[0].fs if segs else 250_000
    for run, rs in sorted(by_run.items()):
        if len(rs) < 2:
            continue
        a, b = rs[0], rs[-1]
        clock = b["stream_time"] - a["stream_time"]
        read = (b["run_index"] - a["run_index"]) / fs
        short = clock - read
        ppm = 1e6 * short / clock if clock > 0 else 0.0
        print("   backend run %d: device clock %.3f s, frames read %.3f s, difference %+.3f s (%+.0f ppm)"
              % (run, clock, read, short, ppm))
        # a block is 8.2 ms and the two readings are taken a few ms apart;
        # anything like an overflow is seconds, and a crystal is tens of ppm
        if abs(short) > 0.05 + 100e-6 * clock:
            fail("COMPLETE", "backend run %d lost %.3f s against the device clock" % (run, short))
    restarts = max((r["capture_restarts"] for r in rows), default=0)
    dropped = max((r["dropped"] for r in rows), default=0)
    print("   record-action restarts %d, blocks dropped for a slow disk %d" % (restarts, dropped))
    if restarts:
        fail("COMPLETE", "the record action died %d time(s)" % restarts)
    if dropped:
        fail("COMPLETE", "%d blocks were dropped before reaching the disk" % dropped)

if "--no-analysis" not in args and segs:
    from friture.recording.analysis_log import log_path, read_log
    fs0 = segs[0].fs
    print("\nanalysis logs:")
    per_run = defaultdict(list)
    run_end = {}
    for s in segs:
        run_end[s.run_id] = max(run_end.get(s.run_id, 0), s.run_index + s.n_frames)
    torn = 0
    for s in segs:
        stem = s.wav[:-4]
        head, recs, bad = read_log(log_path(folder, stem))
        torn += bad
        if head is None:
            if s.n_frames >= s.fs:
                fail("LOGGED", "%s has no analysis log" % s.wav)
            continue
        for r in recs:
            if r["seg"] != stem or not (0 <= r["pos"] < s.n_frames):
                fail("LOGGED", "%s: a record points at %s:%d" % (s.wav, r["seg"], r["pos"]))
                break
            per_run[r["run"]].append(r)
    counts = defaultdict(int)
    for run_id, recs in per_run.items():
        for r in recs:
            counts[r["kind"]] += 1
            if r["idx"] + r.get("n", round(r["dur"] * fs0)) > run_end.get(run_id, 0):
                fail("CONTAINED", "a %s record at %d runs past the end of run %s" % (r["kind"], r["idx"], run_id))
        levels = sorted((r["idx"], r.get("n", round(r["dur"] * fs0))) for r in recs if r["kind"] == "levels")
        for (a, la), (b, _) in zip(levels, levels[1:]):
            if b < a + la:
                fail("COVERED", "run %s: level records overlap at %d" % (run_id, b))
                break
        run_frames = sum(x.n_frames for x in segs if x.run_id == run_id)
        covered = sum(l for _, l in levels)
        share = covered / run_frames if run_frames else 1.0
        print("   run %s: levels cover %.1f%% of %.1f s" % (run_id, 100 * share, run_frames / fs0))
        # everything but a final remnant under MIN_FRACTION of a second
        if run_frames - covered > 0.4 * fs0 + 1:
            fail("COVERED", "run %s: %.2f s has no level record" % (run_id, (run_frames - covered) / fs0))
    print("   records: %s; torn lines skipped %d" % (dict(counts), torn))

    if "--compare-with" in args:
        other = Path(args[args.index("--compare-with") + 1])
        print("\ncomparing with the logs in %s:" % other)
        same = differ = 0
        for s in segs:
            stem = s.wav[:-4]
            _, a, _ = read_log(log_path(folder, stem))
            _, b, _ = read_log(log_path(other, stem))
            # order-insensitive: records come out in the order their windows
            # complete, which depends on how the audio was cut into pieces
            strip = lambda rs: sorted(({k: v for k, v in r.items() if k != "t"} for r in rs),
                                      key=lambda r: (r["idx"], r["kind"]))
            a_s, b_s = strip(a), strip(b)
            if a_s == b_s:
                same += len(a_s)
                continue
            differ += 1
            fail("COMPARE", "%s: %d records here, %d there, first difference at %s"
                 % (s.wav, len(a_s), len(b_s),
                    next((i for i, (x, y) in enumerate(zip(a_s, b_s)) if x != y), min(len(a_s), len(b_s)))))
        print("   %d records identical apart from the timestamp; %d file(s) differ" % (same, differ))

print("\n%s" % ("ALL OK" if not problems else "%d PROBLEM(S)" % len(problems)))
sys.exit(0 if not problems else 1)
