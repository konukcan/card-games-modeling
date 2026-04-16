# Resume Instructions (if the overnight session died)

**Night context:** This is Night 2 of the MCMC review (continuation of Night 1 which terminated early at 6.5/10 after 3 rounds). Night 1 artifacts are archived at `night1-archive/`. Tonight's active state lives under `review-stage/`.

If you come back (from commute, from sleep, from a Wi-Fi drop) and the run appears stopped, use one of these two prompts depending on what you see.

## Scenario A — The `claude` session is still open but stuck on an error

Type this into the existing session:

> The network dropped. Please retry the last action and continue the run. If the retry fails on the same call, fall back per ARIS_LAUNCH_PROMPT.md (Claude sub-agent reviewer). Keep going.

## Scenario B — Terminal is closed or `claude` exited

Open a new terminal:

```bash
cd ~/Documents/self-explanations-project/card-games-modelling/.worktrees/aris-mcmc-review
claude
```

Paste into the new session:

> Resume the overnight MCMC review run. Read ARIS_LAUNCH_PROMPT.md, then review-stage/REVIEW_STATE.json to see where we stopped. Check review-stage/experiments/ for any experiments that completed while the session was dead. Then continue the /auto-review-loop from the next round. Do not restart completed rounds.

## Sanity checks before resuming (optional but fast)

```bash
# See where the run got to
cat review-stage/REVIEW_STATE.json 2>/dev/null || echo "No state yet"

# See what's been committed
git log --oneline aris/mcmc-review-20260411 -10

# See if any experiments are still running (PID files)
for pid_file in review-stage/experiments/*/*.pid; do
  [ -f "$pid_file" ] || continue
  pid=$(cat "$pid_file")
  if ps -p "$pid" > /dev/null 2>&1; then
    echo "STILL RUNNING: $pid_file (pid $pid)"
  else
    echo "finished: $pid_file"
  fi
done
```

## If something looks wrong

- Uncommitted changes from the dying session? They're still in the working tree — the resume prompt instructs the new session to commit them first.
- 24-hour window exceeded? ARIS will start fresh instead of resuming. Edit `review-stage/REVIEW_STATE.json` timestamp to something recent if you want to force resume, OR just start over — the committed work from prior rounds is still preserved in git.
