#!/bin/sh
set -eu
umask 077
cd /home/csaba/repos/AWS/TrainiumFrontierCompetition/trainium_frontier
mkdir -p out/fresh-start-artifact-backup
exec /usr/bin/nice -n 10 /usr/bin/timeout --kill-after=30s 3500s /usr/bin/python3 \
  research/fresh_start_artifact_backup.py >>out/fresh-start-artifact-backup/cron.log 2>&1
