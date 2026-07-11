# Contributing to ShopSage

This document describes the workflow for contributing to this repository. Read it once before your first PR — after that it becomes routine.

---

## 1. Getting Access

This repository uses a collaborator-based workflow (not forks). You must be added as a collaborator by the repo owner before you can push branches. If `git push` fails with a permissions error, you have not yet accepted the collaborator invite — check your email/GitHub notifications, or reach out to the repo owner.

---

## 2. One-Time Setup

```bash
# Clone the repo
git clone <repo-url>
cd ShopSage

# Set up your environment (see README.md for full details)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your own API keys — never commit .env
```

Verify your setup by following the quickstart in `README.md`.

---

## 3. Golden Rule

**Never commit directly to `main`.** All changes go through a feature branch → pull request → review → merge. Everything below is the mechanics of that process.

---

## 4. Everyday Workflow

### Step 1 — Sync with `main` before starting work

```bash
git checkout main
git pull origin main
```

Do this every time you start a work session, even mid-task. It's the single biggest thing that prevents painful merge conflicts later.

### Step 2 — Create a feature branch

Branch names follow the task numbering in `tasks.md`, so anyone can tell what a branch is for at a glance:

```
feature/<task-number>-<short-description>
```

```bash
git checkout -b feature/04-synthetic-dataset
```

Examples:
- `feature/03-system-prompt`
- `feature/04-synthetic-dataset`
- `feature/05-product-catalog`

### Step 3 — Work, commit in small chunks

Commit often — don't wait until an entire task is done to make your first commit. Small, focused commits are easier to review and easier to revert if something breaks.

```bash
git add data/shopper_profiles.csv
git commit -m "Add 20 synthetic shopper profiles with order history"
```

**Commit message style:** short, present tense, describes *what* changed:
- `Add waterproof jacket entries to product catalog`
- `Fix budget filter dropping items exactly at $80`

Avoid vague messages like `updates` or `wip`.

### Step 4 — Push your branch

```bash
git push origin feature/04-synthetic-dataset
```

On the first push of a new branch, Git will suggest the exact command (`git push --set-upstream origin <branch>`) — copy-paste it.

### Step 5 — Open a Pull Request

- **Base:** `main` ← **Compare:** your feature branch
- **Title:** matches the branch's purpose, e.g. "Task 4: Synthetic shopper dataset"
- **Description:** what you did and how to verify it — reuse the "Evidence of Completion" from `tasks.md`
- Request a review before merging, particularly for tasks with cross-team dependencies

### Step 6 — Merge

Before merging, always rebase or merge the latest `main` into your branch to catch conflicts early:

```bash
git checkout main
git pull origin main
git checkout feature/04-synthetic-dataset
git merge main
# resolve any conflicts, then push again
git push origin feature/04-synthetic-dataset
```

Merge the PR using "Squash and merge" to keep `main`'s history clean — one commit per task.

### Step 7 — Clean up

```bash
git checkout main
git pull origin main
git branch -d feature/04-synthetic-dataset
```

---

## 5. Worked Example

```bash
# 1. Sync
git checkout main
git pull origin main

# 2. Branch
git checkout -b feature/04-synthetic-dataset

# 3. Work + commit
git add data/shopper_profiles.csv data/order_history.csv
git commit -m "Add synthetic shopper profiles and order history dataset"

git add docs/dataset_summary.md
git commit -m "Add summary counts of profiles and orders per tasks.md evidence requirement"

# 4. Push
git push origin feature/04-synthetic-dataset

# 5. Open PR: feature/04-synthetic-dataset -> main
#    Title: "Task 4: Synthetic shopper dataset"
#    Description: "20 shopper profiles, 45 orders across profiles. See docs/dataset_summary.md for counts."

# 6. After review/merge, clean up locally
git checkout main
git pull origin main
git branch -d feature/04-synthetic-dataset
```

---

## 6. Troubleshooting

**Permission denied on push**
→ You have not accepted the collaborator invite, or you're authenticated as the wrong GitHub account. Run `git remote -v` to confirm the remote URL, and check which account you're logged into locally.

**Merge conflict**
→ Git marks conflicting sections in the file with `<<<<<<<`, `=======`, `>>>>>>>`. Open the file, resolve manually, remove the conflict markers, then:
```bash
git add <file>
git commit
git push origin <your-branch>
```
If you're unsure which version is correct, raise it with the team before guessing — silently choosing one side can quietly undo someone else's work.

**Accidentally committed to `main`**
→ Do not force-push. Flag it to the team immediately — it's a quick fix if caught early, and much messier once others have pulled `main`.

**Branch is stale relative to `main`**
```bash
git checkout main
git pull origin main
git checkout <your-branch>
git merge main
# resolve conflicts, then push
```

---

## 7. Task-Level Expectations

Every PR should map to a row in `tasks.md` and satisfy its **Definition of Done** and **Evidence of Completion** columns. Include that evidence (screenshot, transcript, log, etc.) directly in the PR description — this keeps a clean audit trail for later retrospectives and demo prep.

If a task grows beyond its estimated scope, keep the PR focused on that one task rather than bundling multiple tasks together.
