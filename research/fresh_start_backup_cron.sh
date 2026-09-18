#!/bin/sh
# Invoked by the user's cron daemon, independently of the Codex GUI.
set -eu
umask 077
cd /home/csaba/repos/AWS/TrainiumFrontierCompetition/trainium_frontier
mkdir -p out/fresh-start-backup
exec /usr/bin/timeout --kill-after=15s 480s /usr/bin/python3 \
  research/fresh_start_backup.py >>out/fresh-start-backup/cron.log 2>&1
