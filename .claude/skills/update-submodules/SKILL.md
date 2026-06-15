---
name: update-submodules
description: Update a vendored git submodule (torchtitan, torchft, or any future fork-based submodule) to a newer upstream version while preserving Panocular's custom changes that live on top of upstream. Use this whenever the user wants to "update torchtitan/torchft", "update the submodules", "pull in upstream changes", "rebase our fork", "bump a submodule", "sync the fork with upstream", or resolve conflicts between our changes and new upstream code. Trigger even if the user only names one submodule or just says "update the submodule".
---

# Update a fork-based submodule from upstream

## Mental model — read this first

This repo (`symphony-learn`) vendors libraries as git **submodules** that point at
**Panocular forks** (e.g. `PanocularAI/torchtitan`, `PanocularAI/torchft`), not at the
upstream projects directly.

Each fork's `main` is structured the same way:

```
  our custom commits           ← Panocular's changes, replayed on top   ┐
  ...                                                                    │  "the stack"
  our custom commits                                                     ┘
  <upstream snapshot HEAD>     ← a commit from the upstream project
  ...upstream history...
```

So our changes are a **small stack of commits replayed on top of an upstream snapshot**.
**Updating = rebase that stack onto a newer upstream commit, resolve conflicts, push the fork,
then move the submodule pointer in `symphony-learn` and commit it.** That's the whole job for
*every* submodule — only the remote URLs, the files our stack touches, and the verification
commands differ. Everything below is the careful, generic version of those five moves.

**Never hardcode the stack's SHAs or commit count** — always rediscover it dynamically per
submodule (Step 2), because the stack grows over time and differs between submodules.

## Submodule registry

Run `cat .gitmodules` from the repo root to see the current set. Known mappings:

| Submodule path | Fork (origin) | Upstream remote to add |
|---|---|---|
| `torchtitan` | `git@github.com:PanocularAI/torchtitan.git` | `https://github.com/pytorch/torchtitan.git` |
| `torchft` | `git@github.com:PanocularAI/torchft.git` | `https://github.com/pytorch/torchft.git` |

If the user says "update the submodules" without naming one, ask which (or do all of them) and
run the steps below **once per submodule**.

## Guardrails

- **Pushing the fork and force-pushing rewrite shared history** — these are outward-facing.
  Confirm before pushing, and strongly prefer a **new branch + PR** over force-pushing `main`
  (Step 6). Never `git push --force`; only `--force-with-lease`.
- Work on a throwaway branch so a botched rebase is `git rebase --abort` + delete-branch, never a
  corrupted `main`.
- Don't run full training/long jobs to "verify" — verify with builds/imports/tests (Step 5).
- Keep the user in the loop at the decision points: target upstream ref, each conflict
  resolution, whether any of our commits are now obsolete, and how to publish.

---

The steps below use `$SM` for the submodule path (e.g. `torchtitan`). Set it once per submodule
and repeat the whole sequence.

## Step 0 — Orient and check preconditions

```bash
cd "$SM"                          # the submodule, from repo root
git status                        # MUST be clean; if dirty, stop and ask the user
git remote -v                     # expect: origin -> PanocularAI/<name>
```

Ensure the upstream remote exists (URL from the registry above), then fetch everything:

```bash
git remote get-url upstream 2>/dev/null \
  || git remote add upstream <UPSTREAM_URL_FROM_REGISTRY>
git fetch origin
git fetch upstream
```

Note the submodule pointer the parent currently records — it may lag `origin/main`:

```bash
git rev-parse HEAD            # what symphony-learn points at
git rev-parse origin/main    # tip of the fork
```

## Step 1 — Decide the target upstream ref

Default to the latest upstream `main` (`upstream/main`). If the user wants stability, offer a
release tag instead (`git tag -l | sort -V | tail`). Show how much is incoming before committing
to the work:

```bash
OLD_BASE=$(git merge-base origin/main upstream/main)
git log --oneline "$OLD_BASE"..upstream/main | wc -l      # how many upstream commits we'd pull in
git log --oneline "$OLD_BASE"..upstream/main | head -40   # a taste of what's new
```

Pin the chosen ref for the rest of the run, e.g. `TARGET=upstream/main` (or a tag / exact SHA).
Prefer recording the resolved SHA so the result is reproducible.

## Step 2 — Identify OUR stack (the commits to replay)

These are the commits on the fork that aren't in upstream — this is the **authoritative** way to
find the stack:

```bash
git log --oneline upstream/main..origin/main
git log --format='%h %an <%ae> %s' upstream/main..origin/main   # inspect authors
```

Sanity-check the list looks like our changes (small, focused on the dirs we customize). Author
email is only a weak hint — it varies (some commits are `@panocular.ai`, others a personal/GitHub
noreply address), so **don't filter by email**; trust the `upstream/main..origin/main` range. If
the range contains commits that clearly aren't ours, or it's surprisingly large/non-linear, stop
and investigate — the fork may have merged upstream in a way that needs a different `--onto` base
or a merge instead of a rebase. Show the user the list and confirm it's the intended set.

## Step 3 — Rebase the stack onto the target

Work on a dated branch built from the current fork tip, then replay only our commits onto the new
upstream base. Using `--onto <target> <old-base>` guarantees we move exactly the stack from
Step 2 and nothing else:

```bash
git checkout -b "${SM}-update-$(date +%Y%m%d)" origin/main
git rebase --onto "$TARGET" "$OLD_BASE"
```

Clean rebase → skip to Step 5. Otherwise → Step 4.

## Step 4 — Resolve conflicts (the part that needs judgment)

The goal is to **preserve the intent of our change while adapting it to upstream's new
API/structure** — not to blindly keep "ours" or "theirs".

For each conflict:
1. `git status` to see conflicted files. (Our stacks tend to touch a small, stable set —
   e.g. torchtitan's `torchtitan/experiments/ft/` and training entry; torchft's reconfiguration
   logic. Confirm by reading the stack's own diffs from Step 2.)
2. Read the upstream side to understand what changed (renamed functions, moved modules, changed
   signatures, config-registry refactors are common upstream).
3. Read our side to understand the behavior we're adding.
4. Reconcile: reapply our behavior on top of upstream's new shape. If upstream **moved/renamed**
   the thing we patched, follow it there. If a file we touched was **deleted upstream**, find
   where the logic went.
5. `git add <files>` then `git rebase --continue`.

Watch for our change becoming **obsolete** — upstream may have landed equivalent functionality.
If a commit is now redundant, raise it with the user; dropping it via `git rebase --skip` (or
editing it down) keeps our stack minimal. **Note anything dropped or materially reworked in the
final summary.**

If it gets messy, `git rebase --abort` returns you to a clean `origin/main` to retry.

## Step 5 — Verify without a full run

Confirm the tree is coherent — this catches the most common rebase breakage (stale imports,
moved symbols). Verification is submodule-specific:

- **Pure-Python submodule (e.g. torchtitan):** from the repo root, using the project venv:
  ```bash
  python -c "import torchtitan; print(torchtitan.__file__)"
  python -c "import torchtitan.experiments.ft.manager, torchtitan.experiments.ft.checkpoint"
  ls tests && python -m pytest tests -q     # if present and quick
  ```
  Also eyeball that `run_train.sh`'s entry points still exist (`MODULE`, `TRAIN_FILE`,
  `CONFIG_NAME`), since upstream refactors config registries periodically. Do NOT launch
  `run_train.sh`.

- **Submodule with a native/Rust extension (e.g. torchft):** importing requires a build. Rebuild
  the extension before the import check (it ships a Rust core via maturin):
  ```bash
  pip install -e ./torchft        # or: (cd torchft && maturin develop)
  python -c "import torchft; print(torchft.__file__)"
  (cd torchft && cargo build 2>/dev/null; ls tests && python -m pytest tests -q)
  ```
  If a full rebuild isn't feasible in this environment, say so explicitly rather than claiming the
  import passed.

Report exactly what you ran and what passed/failed — don't claim verification you didn't do.

## Step 6 — Publish the fork (confirm first)

Show the user `git log --oneline "$TARGET"..HEAD` (our replayed stack on the new base) and the
diff range, then ask how to publish. Preferred, safest path — push a branch and open a PR:

```bash
git push origin HEAD
gh pr create --repo PanocularAI/$SM --base main --head "${SM}-update-$(date +%Y%m%d)" \
  --title "Rebase onto upstream $SM <date/ref>" \
  --body "Replays our stack onto <TARGET>. Conflicts resolved in <files>. <obsolete commits noted>."
```

Only if the user explicitly wants `main` moved directly (understanding it rewrites shared history)
update it with a lease guard:

```bash
git push --force-with-lease origin "${SM}-update-$(date +%Y%m%d):main"
```

## Step 7 — Bump the submodule pointer in symphony-learn

Point the parent at the new commit and commit that change. Use the merged `origin/main` if you
opened+merged a PR, otherwise the branch tip you pushed.

```bash
cd "$SM"
git fetch origin
git checkout <new-sha-or-origin/main>
cd ..
git add "$SM"
git submodule status "$SM"        # confirm the new SHA
git commit -m "Bump $SM submodule to upstream <ref> (rebased stack)"
```

If you updated multiple submodules, you can stage them all and use one commit, or one per
submodule — match the user's preference. Don't push `symphony-learn` or open its PR unless asked.

End with a summary per submodule: old vs new upstream base, the stack after rebase, conflicts
resolved, anything dropped as obsolete, and what was verified.

## Quick reference

| Action | Command |
|---|---|
| Find our stack | `git log upstream/main..origin/main` |
| Old base | `git merge-base origin/main upstream/main` |
| Replay stack | `git rebase --onto <target> $(git merge-base origin/main upstream/main)` |
| Abort | `git rebase --abort` |
| Publish | branch + `gh pr create` (preferred) or `git push --force-with-lease origin <branch>:main` |
| Bump pointer | `git add <submodule> && git commit` in `symphony-learn` |
