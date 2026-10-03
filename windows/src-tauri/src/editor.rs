// File editing for the assistant personality.
//
// Nothing in this module writes a file because the model asked for it. Every
// change is a *proposal*: the model's tools only ever compute a new content and
// hand it back, the island shows a diff card, and the file is written only when
// the user clicks. That is the same rule the approval card follows, and it is
// why there is no path from a model response to a write.
//
// Three guards stand between a proposal and the disk:
//   * scope — a path outside the folder the user dropped a file into is refused
//     outright, before anything is read or proposed;
//   * a content digest taken when the proposal was made and re-checked at apply
//     time, so a file that changed in between (Claude Code, an editor, a sync
//     client) aborts the whole batch instead of being clobbered;
//   * a backup of every file about to change, written before the first write.
//
// Delete is not a delete: it is a write of the empty string, and the old bytes
// are already in the backup. Restore is not a special action either — it is the
// same proposal flow with the backup's contents as the new value.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{json, Value};

use crate::platform;
use crate::settings;

/// How many times the model may chain tools before it has to answer the user.
pub const MAX_TOOL_ROUNDS: usize = 8;
/// A diff card never renders more than this much of either side.
pub const PREVIEW_CHARS: usize = 4_000;
/// Output a proposed command may show before it is cut.
pub const RUN_OUTPUT_LINES: usize = 20;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

// ── Guards ────────────────────────────────────────────────────────────────────

/// Enough to notice that a file is not the one we read: size plus a digest of
/// the bytes. Not a cryptographic hash — this is optimistic concurrency, and
/// `DefaultHasher` is in the standard library.
#[derive(Serialize, Clone, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Guard {
    pub len: u64,
    pub digest: u64,
}

impl Guard {
    fn of(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let mut h = DefaultHasher::new();
        h.write(&bytes);
        Ok(Self { len: bytes.len() as u64, digest: h.finish() })
    }
}

/// "The file changed since you looked at it" — the whole batch is refused, so a
/// proposal can never be half-applied onto a file it was not written for.
pub fn verify(guard: &Guard, path: &Path) -> Result<(), String> {
    match Guard::of(path) {
        Ok(now) if now == *guard => Ok(()),
        Ok(_) => Err(format!(
            "{} changed since the preview — nothing was written. Ask again.",
            path.display()
        )),
        Err(e) => Err(e),
    }
}

// ── Proposals ─────────────────────────────────────────────────────────────────

/// One proposed change to one path. `old` is kept only so the diff card can
/// render it and so a delete can be undone; it is never written.
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Change {
    pub kind: &'static str,
    /// The path that ends up holding the new content. Empty for a delete.
    pub path: String,
    /// The path the content is read from and guarded against. Empty for a create.
    pub from: String,
    pub old: String,
    pub new: String,
    pub guard: Guard,
    /// `new.len()` vs `old.len()` in lines, precomputed so the front end does
    /// not have to split strings it was never given.
    pub added: usize,
    pub removed: usize,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ProposedRun {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: String,
}

impl ProposedRun {
    /// A command with no shell anywhere in it: the program name and every
    /// argument stay separate, so nothing in a folder name is read as syntax.
    pub fn to_command(&self) -> Command {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args);
        cmd.current_dir(&self.cwd);
        let _ = platform::no_console(&mut cmd);
        cmd
    }
}

/// A change set waiting for a click.
///
/// It holds no model state on purpose: the conversation in `Chat` already has
/// the `tool_use` block that proposed it, so an apply needs nothing from here
/// except the bytes and the guard.
#[derive(Clone)]
pub struct Pending {
    pub id: String,
    pub summary: String,
    pub changes: Vec<Change>,
    pub run: Option<ProposedRun>,
}

/// What the island renders for a proposal.
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PendingView {
    pub id: String,
    pub summary: String,
    pub files: Vec<FilePreview>,
    pub run: Option<RunView>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FilePreview {
    pub kind: String,
    /// Basename for the card title; the full path is shown under it.
    pub name: String,
    pub path: String,
    pub old: String,
    pub new: String,
    /// True when either side was cut to fit the card.
    pub truncated: bool,
    pub added: usize,
    pub removed: usize,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RunView {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: String,
}

fn preview(change: &Change) -> FilePreview {
    let clipped = |s: &str| -> (String, bool) {
        if s.chars().count() <= PREVIEW_CHARS {
            return (s.to_string(), false);
        }
        (s.chars().take(PREVIEW_CHARS).collect::<String>(), true)
    };
    let (old, old_cut) = clipped(&change.old);
    let (new, new_cut) = clipped(&change.new);
    let name = Path::new(if change.path.is_empty() { &change.from } else { &change.path })
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "file".into());
    FilePreview {
        kind: change.kind.to_string(),
        name,
        path: change.path.clone(),
        old,
        new,
        truncated: old_cut || new_cut,
        added: change.added,
        removed: change.removed,
    }
}

pub fn view(pending: &Pending) -> PendingView {
    PendingView {
        id: pending.id.clone(),
        summary: pending.summary.clone(),
        files: pending.changes.iter().map(preview).collect(),
        run: pending.run.as_ref().map(|r| RunView {
            program: r.program.clone(),
            args: r.args.clone(),
            cwd: r.cwd.clone(),
        }),
    }
}

fn count_lines(s: &str) -> usize {
    s.lines().count()
}

// ── Session ───────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct Session {
    /// Folders the editor may touch: the parent of every file the user dropped.
    scope: Mutex<Vec<PathBuf>>,
    pending: Mutex<HashMap<String, Pending>>,
    /// Set in tests so an apply never writes into the user's real backups folder.
    /// `None` is the app, which always uses `backups_dir()`.
    backup_root: Option<PathBuf>,
}

impl Session {
    /// Where this session keeps its backups. Only a test ever overrides it.
    fn backup_dir(&self) -> PathBuf {
        self.backup_root.clone().unwrap_or_else(backups_dir)
    }
}

/// What one model-side tool call did, and whether anything needs a card.
pub struct Outcome {
    /// The text to send back as `tool_result`.
    pub result: String,
    pub is_error: bool,
    /// Set once the model has proposed something the user must see. The loop
    /// stops here and hands this to the island.
    pub card: Option<Card>,
}

pub enum Card {
    Changes { summary: String, changes: Vec<Change> },
    Run { summary: String, run: ProposedRun },
}

impl Outcome {
    pub fn ok(text: impl Into<String>) -> Self {
        Self { result: text.into(), is_error: false, card: None }
    }

    pub fn err(text: impl Into<String>) -> Self {
        Self { result: text.into(), is_error: true, card: None }
    }
}

impl Session {
    /// Called when a file lands. Its folder becomes the only place the editor
    /// may write; nothing else opens the session.
    pub fn allow_folder(&self, file: &Path) {
        if let Some(dir) = file.parent().and_then(|d| d.canonicalize().ok()) {
            let mut scope = self.scope.lock().unwrap();
            if !scope.contains(&dir) {
                scope.push(dir);
            }
        }
    }

    /// True once a file has been dropped — which is what picks the personality.
    pub fn has_scope(&self) -> bool {
        !self.scope.lock().unwrap().is_empty()
    }

    pub fn reset(&self) {
        self.scope.lock().unwrap().clear();
        self.pending.lock().unwrap().clear();
    }

    /// The stored proposal with this id, as the island renders it.
    pub fn pending_of(&self, id: &str) -> Option<PendingView> {
        self.pending.lock().unwrap().get(id).map(view)
    }

    /// The real path of `raw`, refusing anything outside the scope.
    ///
    /// A path that does not exist yet is checked through its parent, so
    /// `../../elsewhere/new.txt` is refused for the same reason a write to an
    /// existing file there would be. Symlinks are resolved before the check, so
    /// a link pointing out of the tree cannot smuggle a write through.
    fn resolve(&self, raw: &str) -> Result<PathBuf, String> {
        let path = Path::new(raw);
        if !path.is_absolute() {
            return Err("paths must be absolute.".into());
        }
        // Asked before anything touches the disk: with nothing dropped, the
        // honest answer is that there is nothing to edit, whatever the path.
        if self.scope.lock().unwrap().is_empty() {
            return Err("no file has been dropped yet, so there is nothing to edit.".into());
        }
        let real = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                let parent = path
                    .parent()
                    .and_then(|p| p.canonicalize().ok())
                    .ok_or_else(|| format!("cannot reach {raw}"))?;
                let name = path.file_name().ok_or_else(|| format!("cannot reach {raw}"))?;
                parent.join(name)
            }
        };
        let scope = self.scope.lock().unwrap();
        if scope.iter().any(|root| real.starts_with(root)) {
            Ok(real)
        } else {
            Err(format!("{raw} is outside the folder you dropped the file into."))
        }
    }

    /// Reads a file for the model. Refused outside the scope like everything else.
    pub fn read(&self, raw: &str) -> Outcome {
        match self.resolve(raw) {
            Err(e) => Outcome::err(e),
            Ok(path) => match std::fs::read_to_string(&path) {
                Ok(text) => {
                    let lines = count_lines(&text);
                    Outcome::ok(format!("{lines} lines, {} bytes.\n\n{text}", text.len()))
                }
                Err(e) => Outcome::err(format!("cannot read {}: {e}", path.display())),
            },
        }
    }

    /// Replaces one exact occurrence and proposes the result. Nothing is
    /// written: the new content comes back so the model can check its own work
    /// and try again, and a card goes up for the user.
    pub fn edit(&self, raw: &str, old_string: &str, new_string: &str, summary: &str) -> Outcome {
        let path = match self.resolve(raw) {
            Ok(p) => p,
            Err(e) => return Outcome::err(e),
        };
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => return Outcome::err(format!("cannot read {}: {e}", path.display())),
        };
        let hits = content.matches(old_string).count();
        if hits == 0 {
            return Outcome::err("that text is not in the file — read it again and copy it exactly.");
        }
        if hits > 1 {
            return Outcome::err(format!(
                "that text appears {hits} times — include more surrounding lines so it is unique."
            ));
        }
        let updated = content.replacen(old_string, new_string, 1);
        let guard = match Guard::of(&path) {
            Ok(g) => g,
            Err(e) => return Outcome::err(e),
        };
        let change = Change {
            kind: "edit",
            path: path.to_string_lossy().to_string(),
            from: path.to_string_lossy().to_string(),
            added: count_lines(&new_string),
            removed: count_lines(&old_string),
            old: content,
            new: updated,
            guard,
        };
        let result = change.new.clone();
        Outcome { result, is_error: false, card: Some(Card::Changes { summary: summary.into(), changes: vec![change] }) }
    }

    /// Proposes a whole file's content — the only tool whose output is unbounded,
    /// and the reason the loop checks `stop_reason` before showing a card.
    pub fn write(&self, raw: &str, content: &str, summary: &str) -> Outcome {
        let path = match self.resolve(raw) {
            Ok(p) => p,
            Err(e) => return Outcome::err(e),
        };
        let existing = std::fs::read_to_string(&path).unwrap_or_default();
        if existing == content {
            return Outcome::ok("the file already says exactly that — nothing to change.");
        }
        let guard = Guard::of(&path).unwrap_or(Guard { len: 0, digest: 0 });
        let change = Change {
            kind: if path.exists() { "edit" } else { "create" },
            path: path.to_string_lossy().to_string(),
            from: path.to_string_lossy().to_string(),
            added: count_lines(content),
            removed: count_lines(&existing),
            old: existing,
            new: content.to_string(),
            guard,
        };
        Outcome {
            result: format!("{} lines proposed for {}. Nothing is written yet — the user has to approve it.", count_lines(content), path.display()),
            is_error: false,
            card: Some(Card::Changes { summary: summary.into(), changes: vec![change] }),
        }
    }

    /// Proposes removing a file. The old bytes go to the backup like any other
    /// change, so this is undoable by proposing them back.
    pub fn delete(&self, raw: &str, summary: &str) -> Outcome {
        let path = match self.resolve(raw) {
            Ok(p) => p,
            Err(e) => return Outcome::err(e),
        };
        let old = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => return Outcome::err(format!("cannot read {}: {e}", path.display())),
        };
        let guard = match Guard::of(&path) {
            Ok(g) => g,
            Err(e) => return Outcome::err(e),
        };
        let p = path.to_string_lossy().to_string();
        let change = Change {
            kind: "delete",
            path: String::new(),
            from: p.clone(),
            added: 0,
            removed: count_lines(&old),
            old,
            new: String::new(),
            guard,
        };
        Outcome {
            result: format!("{} is proposed for deletion. It stays on disk until the user approves, and the backup keeps a copy.", p),
            is_error: false,
            card: Some(Card::Changes { summary: summary.into(), changes: vec![change] }),
        }
    }

    /// Proposes a move. Both ends are resolved through the scope, so a rename
    /// cannot carry a file out of the folder the user opened this in.
    pub fn rename(&self, raw_from: &str, raw_to: &str, summary: &str) -> Outcome {
        let from = match self.resolve(raw_from) {
            Ok(p) => p,
            Err(e) => return Outcome::err(e),
        };
        let to = match self.resolve(raw_to) {
            Ok(p) => p,
            Err(e) => return Outcome::err(e),
        };
        if to.exists() {
            return Outcome::err(format!("{} already exists.", to.display()));
        }
        let guard = match Guard::of(&from) {
            Ok(g) => g,
            Err(e) => return Outcome::err(e),
        };
        let change = Change {
            kind: "rename",
            path: to.to_string_lossy().to_string(),
            from: from.to_string_lossy().to_string(),
            added: 0,
            removed: 0,
            old: String::new(),
            new: String::new(),
            guard,
        };
        Outcome {
            result: format!(
                "{} → {} is proposed. Nothing moves until the user approves.",
                from.display(),
                to.display()
            ),
            is_error: false,
            card: Some(Card::Changes { summary: summary.into(), changes: vec![change] }),
        }
    }

    /// Proposes running a program. The command is never handed to a shell: the
    /// name is resolved against $PATH by us, the arguments stay an argument
    /// vector, and the working directory has to be inside the scope.
    ///
    /// The scope guards where the command *works*, not which interpreter runs
    /// it. `cmd.exe` and `python` live outside the dropped folder, so a program
    /// named on $PATH keeps its own path; only a program the model spelled out
    /// as a path has to be inside the scope, or a drop of one file would make
    /// every installed tool unusable.
    pub fn run(&self, program: &str, args: &[String], cwd: &str, summary: &str) -> Outcome {
        let dir = match self.resolve(cwd) {
            Ok(p) if p.is_dir() => p,
            Ok(_) => return Outcome::err(format!("{cwd} is not a folder.")),
            Err(e) => return Outcome::err(e),
        };
        let resolved = if program.contains(['/', '\\']) {
            match self.resolve(program) {
                Ok(p) => p,
                Err(e) => return Outcome::err(e),
            }
        } else {
            match platform::find_on_path(program) {
                Some(p) => p,
                None => return Outcome::err(format!("{program} is not on PATH.")),
            }
        };
        let run = ProposedRun {
            program: resolved.to_string_lossy().to_string(),
            args: args.to_vec(),
            cwd: dir.to_string_lossy().to_string(),
        };
        let joined = std::iter::once(run.program.clone())
            .chain(run.args.iter().cloned())
            .collect::<Vec<_>>()
            .join(" ");
        Outcome {
            result: format!("`{joined}` is proposed. It runs only if the user clicks Run."),
            is_error: false,
            card: Some(Card::Run { summary: summary.into(), run }),
        }
    }

    /// Turns a card into a stored proposal and hands back its id.
    pub fn hold(&self, card: Card) -> String {
        let id = format!("edit-{}", NEXT_ID.fetch_add(1, Ordering::Relaxed));
        let (summary, changes, run) = match card {
            Card::Changes { summary, changes } => (summary, changes, None),
            Card::Run { summary, run } => (summary, Vec::new(), Some(run)),
        };
        self.pending
            .lock()
            .unwrap()
            .insert(id.clone(), Pending { id: id.clone(), summary, changes, run });
        id
    }

    fn take(&self, id: &str) -> Result<Pending, String> {
        self.pending.lock().unwrap().remove(id).ok_or_else(|| "That proposal is no longer on screen.".into())
    }

    /// Forgets a proposal the user threw away, so a stale card cannot apply it.
    pub fn discard(&self, id: &str) {
        self.pending.lock().unwrap().remove(id);
    }

    /// Applies a change set: verify every guard first, back everything up, then
    /// write. The model is told exactly what happened so it can report honestly.
    pub fn apply(&self, id: &str) -> Result<String, String> {
        let pending = self.take(id)?;
        for change in &pending.changes {
            let from = Path::new(&change.from);
            if !from.exists() && change.kind != "create" {
                return Err(format!("{} is gone — nothing was written.", from.display()));
            }
            if from.exists() {
                verify(&change.guard, from)?;
            }
        }
        let mut backup = if pending.changes.is_empty() { None } else { Some(Backup::open(self.backup_root.as_deref())?) };

        let mut notes = Vec::new();
        for (i, change) in pending.changes.iter().enumerate() {
            let from = Path::new(&change.from);
            if let Some(b) = backup.as_mut() {
                b.save(i, change, from)?;
            }
            let target = if change.path.is_empty() { from } else { Path::new(&change.path) };
            match change.kind {
                "rename" => {
                    if let Some(parent) = target.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                    }
                    std::fs::rename(from, target).map_err(|e| format!("cannot move {}: {e}", from.display()))?;
                    notes.push(format!("{} → {}", from.display(), target.display()));
                }
                "delete" => {
                    std::fs::remove_file(from).map_err(|e| format!("cannot delete {}: {e}", from.display()))?;
                    notes.push(format!("deleted {}", from.display()));
                }
                _ => {
                    if let Some(parent) = target.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
                    }
                    write_atomic(target, &change.new)?;
                    notes.push(format!(
                        "{} ({} +{} −{})",
                        target.display(),
                        if change.guard.len == 0 && change.guard.digest == 0 { "new" } else { "changed" },
                        change.added,
                        change.removed
                    ));
                }
            }
        }
        if let Some(b) = backup.as_ref() {
            if let Err(e) = b.close() {
                crate::log::line(format!("editor: manifest not written: {e}"));
            }
        }
        Ok(format!("Applied. {}", notes.join("; ")))
    }

    /// Runs a proposed command. Output is captured with a deadline so a program
    /// that waits for input cannot hang the island.
    ///
    /// A non-zero exit is information for the model, not a failure: the user
    /// clicked Run, so the command was always going to happen.
    pub fn execute(&self, id: &str) -> Result<String, String> {
        let pending = self.take(id)?;
        let Some(run) = pending.run.clone() else {
            return Err("That proposal has no command in it.".into());
        };
        let mut cmd = run.to_command();
        cmd.stdin(std::process::Stdio::null());
        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return Err(format!("cannot start {}: {e}", run.program)),
        };
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(child.wait_with_output());
        });
        let outcome = match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(result) => result,
            Err(_) => return Err("the command was still running after 60 s and was left to finish.".into()),
        };
        let text = match outcome {
            Ok(out) => {
                let mut text = String::from_utf8_lossy(&out.stdout).to_string();
                let err = String::from_utf8_lossy(&out.stderr);
                if !err.trim().is_empty() {
                    text.push('\n');
                    text.push_str(&err);
                }
                format!(
                    "exit {}\n{}",
                    out.status.code().map(|c| c.to_string()).unwrap_or_else(|| "killed".into()),
                    text.trim_end()
                )
            }
            Err(e) => format!("could not read its output: {e}"),
        };
        let mut lines = text.lines();
        let kept: Vec<&str> = lines.by_ref().take(RUN_OUTPUT_LINES).collect();
        let dropped = lines.count();
        let mut shown = kept.join("\n");
        if dropped > 0 {
            shown.push_str(&format!("\n… {dropped} more lines"));
        }
        Ok(shown)
    }

    pub fn restore(&self, backup_id: &str) -> Outcome {
        let dir = self.backup_dir().join(backup_id);
        let manifest = match std::fs::read_to_string(dir.join("manifest.json")) {
            Ok(m) => m,
            Err(_) => return Outcome::err(format!("no changes saved under {backup_id}.")),
        };
        let entries: Vec<Value> = match serde_json::from_str(&manifest) {
            Ok(v) => v,
            Err(e) => return Outcome::err(format!("unreadable backup {backup_id}: {e}")),
        };
        let mut changes = Vec::new();
        for entry in entries {
            let (Some(target), Some(saved)) = (
                entry.get("target").and_then(Value::as_str),
                entry.get("saved").and_then(Value::as_str),
            ) else {
                continue;
            };
            let from = dir.join(saved);
            let Ok(new) = std::fs::read_to_string(&from) else { continue };
            let path = match self.resolve(target) {
                Ok(p) => p,
                Err(e) => return Outcome::err(e),
            };
            let old = std::fs::read_to_string(&path).unwrap_or_default();
            let guard = Guard::of(&path).unwrap_or(Guard { len: 0, digest: 0 });
            let p = path.to_string_lossy().to_string();
            changes.push(Change {
                kind: if old.is_empty() { "create".into() } else { "edit".into() },
                path: p.clone(),
                from: p,
                added: count_lines(&new),
                removed: count_lines(&old),
                old,
                new,
                guard,
            });
        }
        if changes.is_empty() {
            return Outcome::err(format!("{backup_id} holds nothing that can be put back."));
        }
        Outcome {
            result: format!("Restoring {} file(s) from {backup_id}. Nothing is written yet.", changes.len()),
            is_error: false,
            card: Some(Card::Changes {
                summary: format!("Restore {} file(s) to their state before {backup_id}", changes.len()),
                changes,
            }),
        }
    }

    /// Every backup on disk, newest first — what the model reads to find an id.
    pub fn backups(&self) -> Outcome {
        let dir = backups_dir();
        let mut lines = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut found: Vec<(String, usize)> = entries
                .flatten()
                .filter_map(|e| {
                    let id = e.file_name().to_string_lossy().to_string();
                    let count = std::fs::read_dir(e.path()).ok()?.flatten().count();
                    Some((id, count))
                })
                .collect();
            found.sort_by(|a, b| b.0.cmp(&a.0));
            lines = found
                .into_iter()
                .map(|(id, count)| format!("{id} — {count} file(s)"))
                .collect();
        }
        if lines.is_empty() {
            return Outcome::ok("no changes have been applied yet.");
        }
        Outcome::ok(lines.join("\n"))
    }
}

// ── Backups ───────────────────────────────────────────────────────────────────

pub fn backups_dir() -> PathBuf {
    settings::local_dir().join("backups")
}

/// One folder per apply. The old bytes land there before anything is written,
/// and `manifest.json` says which saved file belongs to which target — that is
/// the whole of the undo story.
struct Backup {
    dir: PathBuf,
    entries: Vec<Value>,
}

impl Backup {
    fn open(root: Option<&Path>) -> Result<Self, String> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        // A plain timestamp is not a unique name: apply, drop another file,
        // apply again, and the two batches are seconds apart at most — the
        // second one would land in the first one's folder and overwrite the
        // bytes that made it restorable. The counter only moves on a clash.
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let root = root.map(Path::to_path_buf).unwrap_or_else(backups_dir);
        let dir = loop {
            let n = SEQ.fetch_add(1, Ordering::Relaxed);
            let candidate = root.join(if n == 0 { stamp.to_string() } else { format!("{stamp}-{n}") });
            if !candidate.exists() {
                break candidate;
            }
        };
        std::fs::create_dir_all(&dir).map_err(|e| format!("cannot make a backup folder: {e}"))?;
        Ok(Self { dir, entries: Vec::new() })
    }

    /// Copies the old bytes aside. A file that does not exist yet has nothing
    /// to save, and its entry records `null` so a restore removes it again.
    fn save(&mut self, i: usize, change: &Change, from: &Path) -> Result<(), String> {
        let target = if change.path.is_empty() { &change.from } else { &change.path };
        let name = format!("{i}-{}", Path::new(target).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "file".into()));
        if from.exists() && change.kind != "rename" {
            std::fs::copy(from, self.dir.join(&name)).map_err(|e| format!("cannot back up {}: {e}", from.display()))?;
            self.entries.push(json!({ "target": target, "saved": name }));
        } else {
            self.entries.push(json!({ "target": target, "saved": Value::Null }));
        }
        Ok(())
    }

    fn close(&self) -> std::io::Result<()> {
        std::fs::write(self.dir.join("manifest.json"), serde_json::to_string_pretty(&self.entries).unwrap_or_default())
    }
}

/// Writes through a temporary file in the same folder, so a crash mid-write
/// cannot leave a half-written file where a whole one used to be.
fn write_atomic(path: &Path, content: &str) -> Result<(), String> {
    let tmp = path.with_extension(format!(
        "{}.coucou-tmp",
        path.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default()
    ));
    std::fs::write(&tmp, content).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot replace {}: {e}", path.display())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("coucou-editor-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A session whose scope is the sandbox itself, plus a file inside it.
    ///
    /// The backups root is the sandbox too: an apply in a test must never write
    /// into %LOCALAPPDATA%\Coucou\backups, where it would sit next to the user's
    /// real undo history and break every test that runs after it.
    fn session(dir: &Path, name: &str, body: &str) -> (Session, PathBuf) {
        let file = dir.join(name);
        std::fs::write(&file, body).unwrap();
        let s = Session {
            scope: Mutex::new(Vec::new()),
            pending: Mutex::new(HashMap::new()),
            backup_root: Some(dir.join("backups")),
        };
        s.allow_folder(&file);
        (s, file)
    }

    /// Every backup folder this session has written, oldest first.
    fn backups_of(s: &Session) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(s.backup_dir())
            .expect("no backups were written")
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn a_change_never_touches_the_file_until_it_is_applied() {
        let dir = sandbox("preview");
        let (s, file) = session(&dir, "note.txt", "alpha\nbeta\n");

        let out = s.edit(file.to_str().unwrap(), "beta", "gamma", "fix the typo");
        assert!(!out.is_error, "edit should be accepted: {}", out.result);
        assert!(matches!(out.card, Some(Card::Changes { .. })));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "alpha\nbeta\n");

        let id = s.hold(out.card.unwrap());
        s.apply(&id).unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "alpha\ngamma\n");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_that_changed_underneath_the_proposal_aborts_the_whole_batch() {
        let dir = sandbox("guard");
        let (s, file) = session(&dir, "note.txt", "alpha\n");

        let out = s.edit(file.to_str().unwrap(), "alpha", "beta", "one word");
        let id = s.hold(out.card.unwrap());

        // Someone else — Claude Code, an editor, a sync client — got there first.
        std::fs::write(&file, "written by someone else\n").unwrap();

        let err = s.apply(&id).unwrap_err();
        assert!(err.contains("changed since the preview"), "got: {err}");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "written by someone else\n");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_path_outside_the_dropped_folder_is_refused() {
        let dir = sandbox("scope");
        let outside = sandbox("scope-outside");
        let (s, _file) = session(&dir, "note.txt", "hello");
        let victim = outside.join("secret.txt");
        std::fs::write(&victim, "not yours").unwrap();

        let out = s.edit(victim.to_str().unwrap(), "not yours", "mine now", "...");
        assert!(out.is_error);
        assert!(out.result.contains("outside"), "got: {}", out.result);
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "not yours");

        // …and the same holds for a file that would be created there.
        let fresh = outside.join("new.txt");
        let out = s.write(fresh.to_str().unwrap(), "hello", "...");
        assert!(out.is_error);
        assert!(!fresh.exists());

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn traversal_out_of_the_scope_is_refused_even_though_the_path_looks_local() {
        let dir = sandbox("trav");
        let outside = sandbox("trav-outside");
        let (s, _file) = session(&dir, "note.txt", "hello");
        let sneaky = format!("{}\\..\\{}\\new.txt", dir.display(), outside.file_name().unwrap().to_string_lossy());
        assert!(s.write(&sneaky, "x", "...").is_error);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn an_ambiguous_edit_is_refused_rather_than_guessed_at() {
        let dir = sandbox("ambig");
        let (s, file) = session(&dir, "note.txt", "x\nx\n");

        let out = s.edit(file.to_str().unwrap(), "x", "y", "...");
        assert!(out.is_error);
        assert!(out.result.contains("2 times"), "got: {}", out.result);
        assert!(out.card.is_none(), "a refused edit must not raise a card");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deleting_keeps_the_bytes_and_restores_them_as_an_ordinary_change() {
        let dir = sandbox("delete");
        let (s, file) = session(&dir, "note.txt", "keep me\n");

        let out = s.delete(file.to_str().unwrap(), "it is stale");
        let id = s.hold(out.card.unwrap());
        assert!(file.exists(), "proposing a delete must not remove anything");
        s.apply(&id).unwrap();
        assert!(!file.exists());

        // The backup is a proposal like any other — put the file back through it.
        let stamp = backups_of(&s).pop().expect("no backup folder");
        let out = s.restore(&stamp);
        assert!(!out.is_error, "restore said: {}", out.result);
        let id = s.hold(out.card.unwrap());
        s.apply(&id).unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "keep me\n");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_applies_seconds_apart_keep_their_own_backups() {
        let dir = sandbox("two-batches");
        let (s, first) = session(&dir, "one.txt", "first\n");
        s.apply(&s.hold(s.edit(first.to_str().unwrap(), "first", "1st", "a").card.unwrap())).unwrap();
        s.allow_folder(&dir.join("two.txt"));
        let one = backups_of(&s).pop().expect("no backup for the first apply");

        // A second file in the same folder, proposed and applied a moment later.
        // Both applies land in the same second far more often than not, so the
        // backup names have to stay apart or the first batch's bytes are lost.
        let second = dir.join("two.txt");
        std::fs::write(&second, "second\n").unwrap();
        s.apply(&s.hold(s.edit(second.to_str().unwrap(), "second", "2nd", "b").card.unwrap())).unwrap();

        let folders = backups_of(&s);
        assert_eq!(folders.len(), 2, "each apply needs its own backup folder");
        let two = folders.last().unwrap();

        // Restoring the second batch must not hand back the first batch's bytes.
        let out = s.restore(two);
        assert!(!out.is_error, "the second batch: {}", out.result);
        assert_eq!(
            restored_bytes(out.card.unwrap()),
            "second\n",
            "the newest backup must hold the second batch, not the first"
        );

        // And the first batch is still restorable on its own.
        let out = s.restore(&one);
        assert!(!out.is_error, "the first batch: {}", out.result);
        assert_eq!(restored_bytes(out.card.unwrap()), "first\n");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The bytes a restore proposes putting back — its `new` side, not the `old`
    /// the file happens to hold right now.
    fn restored_bytes(card: Card) -> String {
        match card {
            Card::Changes { changes, .. } => changes[0].new.clone(),
            Card::Run { .. } => String::new(),
        }
    }

    #[test]
    fn a_rename_cannot_carry_a_file_out_of_the_scope() {
        let dir = sandbox("rename");
        let outside = sandbox("rename-outside");
        let (s, file) = session(&dir, "note.txt", "body");

        let out = s.rename(
            file.to_str().unwrap(),
            outside.join("note.txt").to_str().unwrap(),
            "move it",
        );
        assert!(out.is_error);
        assert!(out.result.contains("outside"), "got: {}", out.result);
        assert!(file.exists());

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn a_command_is_proposed_not_run() {
        let dir = sandbox("run");
        let (s, _file) = session(&dir, "note.txt", "hello");
        let marker = dir.join("ran.txt");

        let out = s.run(
            "cmd",
            &[
                "/c".into(),
                format!("echo ran > {}", marker.display()),
            ],
            dir.to_str().unwrap(),
            "check something",
        );
        assert!(!out.is_error, "run should be proposable: {}", out.result);
        assert!(!marker.exists(), "proposing must not run it");

        let id = s.hold(out.card.unwrap());
        let output = s.execute(&id).unwrap();
        assert!(marker.exists(), "clicking Run has to actually run it: {output}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_command_whose_program_is_not_on_path_is_refused() {
        let dir = sandbox("nopath");
        let (s, _file) = session(&dir, "note.txt", "hello");
        let out = s.run("definitely-not-a-real-program-xyz", &[], dir.to_str().unwrap(), "...");
        assert!(out.is_error);
        assert!(out.card.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nothing_is_in_scope_until_a_file_is_dropped() {
        let s = Session::default();
        let out = s.read("C:\\anything\\at\\all.txt");
        assert!(out.is_error);
        assert!(out.result.contains("no file has been dropped"), "got: {}", out.result);
    }

    #[test]
    fn a_proposal_can_only_be_applied_once() {
        let dir = sandbox("once");
        let (s, file) = session(&dir, "note.txt", "alpha\n");
        let out = s.edit(file.to_str().unwrap(), "alpha", "beta", "...");
        let id = s.hold(out.card.unwrap());
        s.apply(&id).unwrap();
        assert!(s.apply(&id).is_err(), "the same card must not apply twice");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_write_leaves_no_temporary_file_behind() {
        let dir = sandbox("atomic");
        let (s, file) = session(&dir, "note.txt", "alpha\n");
        let out = s.edit(file.to_str().unwrap(), "alpha", "beta\nbeta", "...");
        let id = s.hold(out.card.unwrap());
        s.apply(&id).unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("tmp"))
            .collect();
        assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
        // Only the matched text moves: the newline that followed it stays put.
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "beta\nbeta\n");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
