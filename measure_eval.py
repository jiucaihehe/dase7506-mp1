"""Run the unchanged course evaluator and record whole-process peak RSS.

Usage: python measure_eval.py --checkpoint ... --split test --output results/x.json
All arguments are passed unchanged to evaluate.py. CPU scoring time is taken
from that evaluator's JSON, not from process startup or data loading.
"""
import json
from pathlib import Path
import platform
import resource
import runpy
import sys
import time


if __name__ == '__main__':
    if '--output' not in sys.argv:
        raise SystemExit('An explicit --output path is required.')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    started = time.perf_counter()
    runpy.run_path(str(Path(__file__).with_name('evaluate.py')), run_name='__main__')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == 'darwin' else peak * 1024)
    result = dict(peak_rss_bytes=peak_bytes, peak_rss_gib=peak_bytes / 2**30,
                  process_seconds=time.perf_counter()-started,
                  platform=platform.platform(), python=platform.python_version(),
                  measurement='resource.getrusage(RUSAGE_SELF).ru_maxrss; entire evaluator process')
    output.with_suffix('.resources.json').write_text(json.dumps(result, indent=2)+'\n')
