#!/usr/bin/env python3
"""Back up completed checkpoints while independent evaluation continues."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import time

from followup import checkpoint_backup
from preserve import ROOT


def main():
    stop = datetime.datetime.fromisoformat('2026-10-01T07:15:00+00:00')
    while datetime.datetime.now(datetime.timezone.utc) < stop:
        tasks = []
        for request in (ROOT/'deadline/requests').glob('request-*.json'):
            m = json.loads(request.read_text())
            if '--variant' not in m['command']:
                continue
            label = m['label']
            receipt = ROOT/'research/run-logs'/(label+'.checkpoint-backup.json')
            if receipt.exists() and json.loads(receipt.read_text()).get('verified'):
                continue
            log = ROOT/'research/run-logs'/(label+'.log')
            if not log.exists():
                continue
            markers = [line.partition('DEADLINE_TRAIN_COMPLETE ')[2]
                       for line in log.read_text(errors='replace').splitlines()
                       if line.startswith('DEADLINE_TRAIN_COMPLETE ')]
            if not markers:
                continue
            variant = m['command'][m['command'].index('--variant')+1]
            tasks.append((m['host'], variant, label, json.loads(markers[-1])))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for f in concurrent.futures.as_completed([pool.submit(checkpoint_backup, *t) for t in tasks]):
                try:
                    f.result()
                except Exception as error:
                    print(json.dumps(dict(checkpoint_backup_error=repr(error))), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
