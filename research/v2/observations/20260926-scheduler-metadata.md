# Read-only scheduler diagnostic — 26 September17:45IST

Observed via sqlite3 URI mode=ro and PRAGMA query_only=ON. Only the three
Trainium task IDs, scheduling/status fields and four most-recent controller
run metadata rows were queried. No prompts/messages/credentials were read;
no app database writes or new task occurred.

Source: /home/poojith-devan/.codex/sqlite/codex-dev.db, automations table.

| ID | Database status | Database RRULE | last_run_at |
|---|---|---|---|
| trainium-safety-gated-controller | ACTIVE | FREQ=MINUTELY;INTERVAL=5 | null |
| trainium-04-00-ist-self-audit | ACTIVE | FREQ=DAILY;BYHOUR=4;BYMINUTE=0;BYSECOND=0;COUNT=1 | null |
| trainium-07-45-ist-morning-report | ACTIVE | FREQ=DAILY;BYHOUR=7;BYMINUTE=45;BYSECOND=0;COUNT=1 | null |

All three next_run_at values are1790417273704 (2026-09-26T10:07:53.704Z),
already in the past. Controller updated_at1790416520951 (09:55:20.951Z).
Controller target_thread_id01a0c3ee-db8c-7350-ac40-9610b4da7be0; cwds[].
automation_runs has no rows for this controller. The separately read TOML
has interval25 and modified16:51IST; the two older task TOMLs were savedPAUSED.

Conclusion: local TOML and database metadata disagree. An actual25-minute
tick remains UNVERIFIED. Absence of local run rows does not prove no heartbeat
ever fired, or establish which store the running app currently uses.
Do not describe the saved-file change as a verified runtime configuration.

Official documentation supports editing schedules/reviewing runs in Scheduled,
including minute intervals in a chat. It does not establish that these direct
TOML edits were applied by this app instance:
https://learn.chatgpt.com/docs/automations

No scheduling-management tool or native-app control is exposed in this session.
Keep direct guarded host handoffs and original lock checks; no second controller
or unsupported app-database patch. Later UI verification should set the existing
controller25minutes, pause the two obsolete tasks and observe an actual tick.
This is a non-host blocker; current research and manual commands continue.
