"""Sequential CPU/FP32 measurements after freezing; no tuning occurs here."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile
from common import sha


def main():
    frozen = json.loads(Path('artifacts/FREEZE.json').read_text())
    for filename, digest in frozen['code_sha256'].items():
        if sha(Path(filename)) != digest:
            raise SystemExit(f'Frozen source changed: {filename}')
    for filename, digest in frozen['checkpoint_sha256'].items():
        if sha(Path('artifacts')/filename) != digest:
            raise SystemExit(f'Frozen checkpoint changed: {filename}')
    root = Path('results')
    for filename,digest in frozen.get('asset_sha256',{}).items():
        if sha(Path(filename))!=digest:
            raise SystemExit(f'Frozen inference asset changed: {filename}')
    root.mkdir(exist_ok=True)
    jobs = [('baseline',i) for i in range(3)] + [('final',i) for i in range(3)]
    # Alternate baseline and final to reduce monotonic load/thermal bias.
    jobs = [item for pair in zip(jobs[:3],jobs[3:]) for item in pair]
    jobs += [(name,0) for name in frozen['ablation_names']]
    records = []
    for name, repeat in jobs:
        output = root/f'{name}-test-r{repeat}.json'
        if output.exists():
            raise SystemExit(f'Refusing to overwrite measurements: {output}')
        command = [sys.executable,'measure_eval.py','--checkpoint',f'artifacts/{name}.pt',
                   '--device','cpu','--precision','fp32','--threads','4',
                   '--split','test','--output',str(output)]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
        score = json.loads(output.read_text())
        memory = json.loads(output.with_suffix('.resources.json').read_text())
        if score['targets'] != 428405 or score['utf8_bytes'] != 1292013:
            raise SystemExit('Incorrect test coverage.')
        row = dict(name=name, repeat=repeat, bpb=score['bpb'], seconds=score['seconds'],
                   peak_rss_gib=memory['peak_rss_gib'])
        records.append(row)
        print(json.dumps(row), flush=True)
    median = {name:statistics.median(r['seconds'] for r in records if r['name']==name)
              for name in ['baseline','final']}
    assets = {}
    for name in frozen['inference_assets']:
        if name.endswith('.npz'):
            with zipfile.ZipFile(name) as archive:
                assets[name]=sum(info.file_size for info in archive.infolist())
        else:
            assets[name]=Path(name).stat().st_size
    memory = max(r['peak_rss_gib'] for r in records if r['name']=='final')
    ratio = median['final']/median['baseline']
    result = dict(records=records, median_cpu_seconds=median, cpu_time_ratio=ratio,
                  final_peak_rss_gib=memory, inference_asset_bytes=assets,
                  inference_assets_mib=sum(assets.values())/2**20,
                  limits_passed=dict(cpu=ratio<=5, ram=memory<=4, assets=sum(assets.values())<=64*2**20),
                  threads=4, precision='fp32',
                  note='Fresh evaluator process each time; scorer seconds exclude imports/loading. Whole-process RSS includes imports and loading. No training runs concurrently.')
    (root/'benchmark.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not all(result['limits_passed'].values()):
        raise SystemExit('A resource limit failed; report this without retuning on test scores.')


if __name__ == '__main__':
    main()
