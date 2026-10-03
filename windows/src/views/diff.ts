// The diff card — what the user looks at before anything is written.
//
// This is the whole safety story in one view: the model proposes, the card shows
// the exact before and after, and the only way a byte changes is a click on
// Apply. There is no path from a tool call to a write that skips this file.

import { h, clear } from "./dom";
import { Bridge, type FilePreview, type PendingView } from "../core/bridge";
import { Sound } from "../core/sound";
import { State } from "../core/state";
import type { ViewActions, ViewHost } from "./views";

const KIND_LABEL: Record<string, string> = {
  edit: "Edit",
  create: "New file",
  delete: "Delete",
  rename: "Rename",
};

const KIND_COLOR: Record<string, string> = {
  edit: "#7C5CFF",
  create: "#22C55E",
  delete: "#F4505E",
  rename: "#F5A524",
};

/**
 * The two sides, side by side. Not a real line diff: the model already knows
 * which lines it changed, and drawing every untouched line around them would
 * bury the two or three that actually matter.
 */
function side(text: string, tone: "old" | "new"): HTMLElement {
  const box = h("div", { class: `diff-side ${tone}` });
  if (text === "") {
    box.append(h("span", { class: "diff-empty", text: tone === "new" ? "(empty)" : "— nothing yet —" }));
    return box;
  }
  // textContent, never innerHTML: a file's contents are the model's own output
  // and must not be able to become markup in the island.
  box.append(h("pre", { class: "diff-text", text }));
  return box;
}

function fileBlock(file: FilePreview): HTMLElement {
  const body =
    file.kind === "rename"
      ? h("div", { class: "diff-note", text: "Renamed. The file moves, its contents untouched." })
      : file.kind === "delete"
        ? h("div", {
            class: "diff-note",
            text: "Removed. The bytes go to the backup first, so this can be proposed back.",
          })
        : h("div", { class: "diff-pair" }, side(file.old, "old"), side(file.new, "new"));

  return h(
    "div",
    { class: "diff-file" },
    h(
      "div",
      { class: "diff-head" },
      h("span", {
        class: "diff-kind",
        style: `color:${KIND_COLOR[file.kind] ?? "#7C5CFF"}`,
        text: KIND_LABEL[file.kind] ?? file.kind,
      }),
      h("span", { class: "diff-name", text: file.name }),
      file.added > 0 ? h("i", { class: "add", text: `+${file.added}` }) : null,
      file.removed > 0 ? h("i", { class: "del", text: `−${file.removed}` }) : null,
      file.truncated ? h("span", { class: "diff-cut", text: "preview cut" }) : null,
    ),
    body,
  );
}

function runBlock(run: NonNullable<PendingView["run"]>): HTMLElement {
  return h(
    "div",
    { class: "diff-file" },
    h(
      "div",
      { class: "diff-head" },
      h("span", { class: "diff-kind", style: "color:#F5A524", text: "Command" }),
      h("span", { class: "diff-name", text: run.program.split(/[\\/]/).pop() ?? run.program }),
    ),
    h("div", { class: "diff-note", text: `working folder: ${run.cwd}` }),
    h("pre", { class: "diff-cmd", text: [run.program, ...run.args].join(" ") }),
  );
}

export function buildDiff(actions: ViewActions): ViewHost {
  const summary = h("div", { class: "title diff-title" });
  const files = h("div", { class: "diff-files" });
  const output = h("pre", { class: "diff-output" });
  const row = h("div", { class: "actions" });

  const el = h(
    "div",
    { class: "view diff-view" },
    h(
      "div",
      { class: "card wash diff-card" },
      h("div", { class: "diff-body" }, summary, files, output, row),
    ),
  );
  (el.querySelector(".card") as HTMLElement).style.setProperty("--wash", "rgba(124,92,255,0.5)");

  let busy = false;
  let shownId: string | null = null;

  /** Both click paths end the same way: the id is gone from both sides. */
  function finish(text: string, failed: boolean) {
    State.runOutput = text;
    State.runFailed = failed;
    State.pending = null;
    busy = false;
    State.notify();
    // The card keeps its size: the proposal id is gone, so the file count that
    // set the height no longer applies.
    actions.requestHeight();
  }

  async function apply() {
    const pending = State.pending;
    if (!pending || busy) return;
    busy = true;
    State.notify();
    try {
      finish(await Bridge.editorApply(pending.id), false);
      Sound.play("approve");
    } catch (err) {
      finish(String(err).replace(/^Error:\s*/, ""), true);
      Sound.play("error");
    }
  }

  async function runIt() {
    const pending = State.pending;
    if (!pending?.run || busy) return;
    busy = true;
    State.runOutput = null;
    State.notify();
    try {
      finish(await Bridge.editorRun(pending.id), false);
      Sound.play("approve");
    } catch (err) {
      finish(String(err).replace(/^Error:\s*/, ""), true);
      Sound.play("error");
    }
  }

  function close() {
    const id = State.pending?.id;
    State.pending = null;
    State.runOutput = null;
    State.runFailed = false;
    if (id) void Bridge.editorDiscard(id);
    Sound.play("blip");
    State.notify();
    actions.requestHeight();
  }

  return {
    el,
    sync() {
      const pending = State.pending;

      if (pending?.id !== shownId) {
        shownId = pending?.id ?? null;
        clear(files);
        if (pending) {
          if (pending.run) files.append(runBlock(pending.run));
          else for (const f of pending.files) files.append(fileBlock(f));
        }
        clear(output);
        output.classList.remove("on", "bad");
        output.textContent = "";
      }

      summary.textContent = pending
        ? pending.summary
        : State.runFailed
          ? "Nothing was changed."
          : "Done.";

      // The output of the run, or the reason it stopped. Nothing else goes
      // here: the model's own text stays in the chat, where it belongs.
      if (State.runOutput != null && output.textContent !== State.runOutput) {
        output.textContent = State.runOutput;
        output.classList.add("on");
        output.classList.toggle("bad", State.runFailed);
      }

      // Buttons are built once and never rebuilt between a mouse-down and a
      // mouse-up — the same rule the approval card follows.
      if (row.dataset.built) return;
      row.dataset.built = "1";
      clear(row);

      if (!pending) {
        row.append(
          h("button", {
            class: "btn primary",
            text: State.runFailed ? "Close" : "Back to chat",
            onclick: () => {
              // Nothing is waiting for an answer any more, so the island may close
              // on its idle timer again. Without this the pin set when the card
              // arrived would outlive it and the island would never auto-close.
              actions.setPin(false);
              State.runOutput = null;
              State.runFailed = false;
              State.notify();
              State.view = "prompt";
            },
          }),
        );
        return;
      }
      const isRun = pending.run != null;
      row.append(
        h("button", { class: "btn secondary", text: "Discard", onclick: close }),
        isRun
          ? h("button", { class: "btn primary", text: "Run it", onclick: () => void runIt() })
          : h("button", {
              class: "btn primary",
              text: pending.files.length > 1 ? `Apply ${pending.files.length} files` : "Apply",
              onclick: () => void apply(),
            }),
      );
    },
  };
}