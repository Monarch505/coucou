// Claude API client — the same integration as ClaudeService.swift: multi-turn
// chat with web search, and files sent as document/image/text blocks.
//
// Everything happens here rather than in the island: the API key never leaves
// the Credential Manager, and file bytes never cross the IPC boundary.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::editor;
use crate::providers;
use crate::secrets;
use crate::settings::Settings;

const ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Server-side fallback: on a policy decline the API retries the same request on
/// a fallback model inside the same call, so the island never shows a dead end.
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const MAX_TOKENS: u32 = 4096;
/// Text and code files are inlined; anything larger is skipped, as on macOS.
const MAX_INLINE_TEXT: u64 = 200_000;

pub const DEFAULT_MODEL: &str = "claude-opus-5";

const SYSTEM_PROMPT: &str = "You are Mochi, a personal AI assistant living at the top of the user's screen. \
You have web search access and can help with absolutely anything — research, coding, finding places, recommendations, tasks, questions. \
Respond in the user's language. Be thorough and complete — use as much detail as the task requires. \
No markdown formatting (no **, no ##, no bullet dashes). Use plain text with line breaks.";

/// The second personality. It is sent fresh on every request, so it can never
/// collide with Mochi's — the two are never in the same conversation.
const EDITOR_SYSTEM_PROMPT: &str = "\
You are the editing half of Mochi, working inside a small app at the top of the user's screen. \
You are given one folder — the folder a file was dropped into — and you may only touch files inside it. \
Other paths are refused by the app, so never even propose them.

HOW TO CHANGE A FILE: prefer edit_file. It replaces one exact snippet, costs almost nothing, and works no \
matter how long the file is. Never read a whole file only to write it back: the output limit would cut it \
and you would leave the file broken. Use write_file only for a file that does not exist yet, or when the \
change genuinely rewrites every line.

TOOLS: read_file, edit_file, write_file, delete_file, rename_file, run_command, list_backups, restore_backup.

BEFORE YOU PROPOSE: read what you are changing, make the smallest change that does what was asked, and \
say in one line what the change is. The user sees a card with a diff and decides. Nothing you propose is \
written until they click.

IF AN EDIT FAILS: the text you passed was not found, or appears more than once. Re-read the file and copy \
the snippet exactly, with enough lines around it to be unique. Do not guess.

IF A FILE CHANGED UNDER YOU: the user was told nothing was written. Read it again before proposing.

TONE: plain text, the user's language, no markdown formatting (no **, no ##, no bullet dashes). \
Say what you changed and what is left to do. No praise, no filler.";

/// Which half of the assistant answers a turn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Persona {
    /// Mochi, as it has always been: web search and conversation.
    Mochi,
    /// The editor: Mochi's tone, with the file tools and no web search.
    Editor,
}

impl Persona {
    fn system_prompt(self) -> &'static str {
        match self {
            Persona::Mochi => SYSTEM_PROMPT,
            Persona::Editor => EDITOR_SYSTEM_PROMPT,
        }
    }

    /// Mochi keeps exactly the one tool it has always had. The editor gets no
    /// web search — searching the web and writing files in the same turn is
    /// the shape of a prompt-injection accident.
    fn tools(self) -> Value {
        match self {
            Persona::Mochi => json!([{ "type": "web_search_20260209", "name": "web_search", "max_uses": 5 }]),
            Persona::Editor => json!(editor_tools()),
        }
    }
}

fn editor_tools() -> Vec<Value> {
    vec![
        json!({
            "name": "read_file",
            "description": "Read a file inside the folder you were given. Returns the whole content with a line count.",
            "input_schema": {
                "type": "object",
                "properties": { "path": { "type": "string", "description": "Absolute path." } },
                "required": ["path"],
            },
        }),
        json!({
            "name": "edit_file",
            "description": "Propose replacing one exact snippet of a file. Nothing is written — the user reviews a diff and clicks. Use this for every change to an existing file, whatever its size. The snippet must appear exactly once; if it appears more than once, include more surrounding lines.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path of the file to change." },
                    "old_string": { "type": "string", "description": "The exact text to replace, unique in the file." },
                    "new_string": { "type": "string", "description": "What replaces it." },
                    "summary": { "type": "string", "description": "One line the user sees on the card." },
                },
                "required": ["path", "old_string", "new_string"],
            },
        }),
        json!({
            "name": "write_file",
            "description": "Propose a file's entire content. Only for a file that does not exist yet, or a change that really does rewrite every line. For anything smaller use edit_file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path." },
                    "content": { "type": "string", "description": "The whole new content." },
                    "summary": { "type": "string", "description": "One line the user sees on the card." },
                },
                "required": ["path", "content"],
            },
        }),
        json!({
            "name": "delete_file",
            "description": "Propose removing a file. The old bytes go to a backup first, so this can be undone by proposing restore_backup with this change's id.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path." },
                    "summary": { "type": "string", "description": "One line the user sees on the card." },
                },
                "required": ["path"],
            },
        }),
        json!({
            "name": "rename_file",
            "description": "Propose moving a file. Both ends must stay inside the folder you were given.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "from": { "type": "string", "description": "Absolute path to move." },
                    "to": { "type": "string", "description": "Absolute path to move it to. Must not exist." },
                    "summary": { "type": "string", "description": "One line the user sees on the card." },
                },
                "required": ["from", "to"],
            },
        }),
        json!({
            "name": "run_command",
            "description": "Propose running a program. No shell is involved: the name is looked up on PATH and the arguments stay separate. Nothing runs until the user clicks Run. stdout and stderr come back so you can report what happened.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "program": { "type": "string", "description": "Program name on PATH, or an absolute path inside the folder." },
                    "args": { "type": "array", "items": { "type": "string" }, "description": "Arguments, passed verbatim." },
                    "cwd": { "type": "string", "description": "Absolute path of an existing folder inside the folder you were given." },
                    "summary": { "type": "string", "description": "One line the user sees on the card." },
                },
                "required": ["program", "cwd"],
            },
        }),
        json!({
            "name": "list_backups",
            "description": "List the changes already applied, newest first, with the id needed by restore_backup.",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        }),
        json!({
            "name": "restore_backup",
            "description": "Propose putting a set of files back to how they were before an applied change. It is a proposal like any other — the user reviews a diff and clicks.",
            "input_schema": {
                "type": "object",
                "properties": { "backup_id": { "type": "string", "description": "The id from list_backups." } },
                "required": ["backup_id"],
            },
        }),
    ]
}

#[derive(Default)]
pub struct Chat {
    /// Full multi-turn history, including tool_use / tool_result blocks.
    messages: Mutex<Vec<Value>>,
}

impl Chat {
    pub fn reset(&self) {
        self.messages.lock().unwrap().clear();
    }

    fn is_empty(&self) -> bool {
        self.messages.lock().unwrap().is_empty()
    }

    fn push(&self, message: Value) {
        self.messages.lock().unwrap().push(message);
    }

    fn pop(&self) {
        self.messages.lock().unwrap().pop();
    }

    fn snapshot(&self) -> Vec<Value> {
        self.messages.lock().unwrap().clone()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ChatContext {
    File { name: String, path: String },
    Window { app_name: String, title: String, url: Option<String> },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatReply {
    pub text: String,
    /// Set when the turn ended in a proposal instead of a reply. The island
    /// switches to the diff card; nothing has been written.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending: Option<editor::PendingView>,
}

/// One chat turn. Returns the assistant's text, or a message the island shows
/// in the note view.
pub async fn send(
    chat: &Chat,
    settings: &Settings,
    session: &editor::Session,
    query: String,
    context: Option<ChatContext>,
) -> Result<ChatReply, String> {
    let model = settings.model.clone();

    // A configured provider takes over completely — its own base URL and its own
    // key. Only when there is none does the Anthropic key matter, so a router
    // user never has to paste one.
    let target: providers::Target = match providers::resolve(settings)? {
        Some(provider) => providers::Target::Provider(provider),
        None => providers::Target::Anthropic {
            url: ENDPOINT.to_string(),
            key: secrets::get("anthropic-api-key")
                .ok_or_else(|| "API key missing. Open settings.".to_string())?,
        },
    };

    // Which half answers depends on whether a file was dropped: with one in
    // hand the editor is the useful half, and web search would only hand it
    // text from the internet to paste into a local file.
    let persona = if session.has_scope() {
        Persona::Editor
    } else {
        Persona::Mochi
    };

    let mut content: Vec<Value> = Vec::new();

    // File / window context rides along with the first message only, exactly
    // like ClaudeService.chat().
    if chat.is_empty() {
        match &context {
            Some(ChatContext::File { name, path }) => {
                // A file we could not inline is said out loud rather than dropped
                // in silence: the model used to look at a "File: notes.txt" line
                // and answer from a file it had never been given.
                match file_block(path) {
                    Ok(block) => content.push(block),
                    Err(why) => content.push(json!({
                        "type": "text",
                        "text": format!("File: {name} — not attached, {why}. Say so rather than guessing its contents."),
                    })),
                }
                // The path the editor works on, not the inbox copy it reads.
                if persona == Persona::Editor {
                    content.push(json!({
                        "type": "text",
                        "text": format!(
                            "Folder to work in: {folder}\nThe user dropped: {name}",
                            folder = std::path::Path::new(path)
                                .parent()
                                .map(|p| p.to_string_lossy().to_string())
                                .unwrap_or_else(|| path.clone()),
                        )
                    }));
                }
                content.push(json!({ "type": "text", "text": format!("File: {name}") }));
            }
            Some(ChatContext::Window { app_name, title, url }) => {
                let mut text = format!("Context — App: {app_name}, Window: {title}");
                if let Some(url) = url {
                    text.push_str(&format!(", URL: {url}"));
                }
                content.push(json!({ "type": "text", "text": text }));
            }
            None => {}
        }
    }
    content.push(json!({ "type": "text", "text": query }));

    chat.push(json!({ "role": "user", "content": content }));

    let reply = converse(&target, chat, &model, persona, Some(session)).await;
    match reply {
        Ok(r) => Ok(r),
        Err(err) => {
            chat.pop(); // keep the history consistent with what the model saw
            Err(err)
        }
    }
}

/// One turn with the model, and — for the editor — any chain of tool calls it
/// makes before it has an answer.
///
/// Mochi makes exactly one request, exactly as before. The editor loops: a
/// `tool_use` block is answered locally and sent back as a `tool_result`, up to
/// [`editor::MAX_TOOL_ROUNDS`], and stops early the moment a proposal comes up,
/// because the user has to see that first.
async fn converse(
    target: &providers::Target,
    chat: &Chat,
    model: &str,
    persona: Persona,
    session: Option<&editor::Session>,
) -> Result<ChatReply, String> {
    for round in 0..=editor::MAX_TOOL_ROUNDS {
        let body = json!({
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": persona.system_prompt(),
            "tools": persona.tools(),
            "fallbacks": "default",
            "messages": chat.snapshot(),
        });

        let response = call(target, &body).await?;

        // A policy decline comes back as HTTP 200 with stop_reason "refusal".
        if response.get("stop_reason").and_then(Value::as_str) == Some("refusal") {
            let why = response
                .get("stop_details")
                .and_then(|d| d.get("explanation"))
                .and_then(Value::as_str)
                .unwrap_or("Claude declined this one.");
            return Err(why.to_string());
        }

        let Some(blocks) = response.get("content").and_then(Value::as_array).cloned() else {
            return Err("Unexpected API response.".into());
        };

        // Store the whole content — tool_use / tool_result blocks included — so
        // the next turn has the right context.
        chat.push(json!({ "role": "assistant", "content": blocks.clone() }));

        let uses: Vec<&Value> = blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
            .collect();

        // Mochi has no local tools, so this is where it always ends.
        let Some(session) = session else {
            return finish(blocks, None);
        };

        if uses.is_empty() {
            return finish(blocks, None);
        }

        // A truncated `write_file` proposal is the one thing that can arrive
        // half-built: the whole content is the output, so hitting the token
        // ceiling means the card would hold a mangled file. Refuse it outright.
        let cut = response.get("stop_reason").and_then(Value::as_str) == Some("max_tokens");
        let mut results = Vec::new();
        let mut card = None;
        for call in uses {
            let (id, name) = match tool_identity(call) {
                Some(v) => v,
                None => continue,
            };
            let input = call.get("input").cloned().unwrap_or_else(|| json!({}));
            let outcome = dispatch(session, &name, &input);
            results.push(json!({
                "type": "tool_result",
                "tool_use_id": id,
                "content": outcome.result,
                "is_error": outcome.is_error,
            }));
            if card.is_none() {
                if let Some(c) = outcome.card {
                    if cut {
                        results.clear();
                        results.push(json!({
                            "type": "tool_result",
                            "tool_use_id": id,
                            "content": "That answer was cut off at the token limit, so the file would have been written incomplete. Nothing was proposed — make the change in smaller pieces, or with edit_file.",
                            "is_error": true,
                        }));
                        break;
                    }
                    card = Some(c);
                }
            }
        }

        // The model's text alongside a proposal is what the card is titled with.
        let text = blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();

        if let Some(card) = card {
            let id = session.hold(card);
            return Ok(ChatReply {
                text: if text.is_empty() {
                    String::new()
                } else {
                    text
                },
                pending: session.pending_of(&id),
            });
        }

        if round == editor::MAX_TOOL_ROUNDS {
            results.push(json!({
                "type": "tool_result",
                "tool_use_id": "none",
                "content": "That was the last tool round. Answer the user now.",
                "is_error": true,
            }));
        }
        chat.push(json!({ "role": "user", "content": results }));
    }
    Err("The assistant went too many rounds without answering.".into())
}

/// The model's closing words, or the card if it proposed something instead.
fn finish(blocks: Vec<Value>, pending: Option<editor::PendingView>) -> Result<ChatReply, String> {
    let text = blocks
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    if text.is_empty() && pending.is_none() {
        return Err("No response text.".into());
    }
    Ok(ChatReply { text, pending })
}

fn tool_identity(call: &Value) -> Option<(String, String)> {
    let id = call.get("id").and_then(Value::as_str)?.to_string();
    let name = call.get("name").and_then(Value::as_str)?.to_string();
    Some((id, name))
}

fn text_arg(input: &Value, key: &str) -> String {
    input.get(key).and_then(Value::as_str).unwrap_or_default().to_string()
}

/// Runs one tool call. Unknown names get an error back rather than being
/// ignored: an unrecognised call with no result would leave the API waiting.
fn dispatch(session: &editor::Session, name: &str, input: &Value) -> editor::Outcome {
    let summary = {
        let s = text_arg(input, "summary");
        if s.is_empty() { String::new() } else { s }
    };
    match name {
        "read_file" => session.read(&text_arg(input, "path")),
        "edit_file" => session.edit(
            &text_arg(input, "path"),
            &text_arg(input, "old_string"),
            &text_arg(input, "new_string"),
            &summary,
        ),
        "write_file" => session.write(&text_arg(input, "path"), &text_arg(input, "content"), &summary),
        "delete_file" => session.delete(&text_arg(input, "path"), &summary),
        "rename_file" => session.rename(&text_arg(input, "from"), &text_arg(input, "to"), &summary),
        "run_command" => {
            let args = input
                .get("args")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect::<Vec<_>>())
                .unwrap_or_default();
            session.run(&text_arg(input, "program"), &args, &text_arg(input, "cwd"), &summary)
        }
        "list_backups" => session.backups(),
        "restore_backup" => session.restore(&text_arg(input, "backup_id")),
        _ => editor::Outcome::err(format!("there is no tool called {name}.")),
    }
}

async fn call(target: &providers::Target, body: &Value) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .build()
        .map_err(|e| e.to_string())?;

    let mut request = client
        .post(target.endpoint())
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("content-type", "application/json")
        .json(&target.prepare_body(body));

    match target {
        providers::Target::Anthropic { key, .. } => {
            request = request
                .header("x-api-key", key)
                .header("anthropic-beta", FALLBACK_BETA);
        }
        providers::Target::Provider(provider) => {
            for (name, value) in providers::headers(provider) {
                request = request.header(name, value);
            }
        }
    }

    let response = request
        .send()
        .await
        .map_err(|e| format!("Network error: {e}"))?;

    let status = response.status();
    let text = response.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        // Surface the API's own message, which is what makes a bad key obvious.
        let detail = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| text.chars().take(200).collect());
        return Err(format!("Claude API {status}: {detail}"));
    }

    // A provider that ignores `"stream": false` still answers in SSE. Take the
    // last text block rather than failing the turn — measured against the local
    // router on 2026-10-01, which streams unless asked not to.
    if text.trim_start().starts_with('{') {
        return serde_json::from_str(&text).map_err(|e| format!("Bad API response: {e}"));
    }

    let streamed: String = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .filter_map(|payload| serde_json::from_str::<Value>(payload.trim()).ok())
        .filter_map(|event| {
            event
                .get("delta")
                .and_then(|d| d.get("text"))
                .or_else(|| event.get("content_block"))
                .and_then(|c| c.get("text"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect();
    if streamed.is_empty() {
        return Err("Bad API response: no JSON and no text in the reply.".into());
    }
    Ok(json!({
        "content": [{ "type": "text", "text": streamed }],
        "stop_reason": "end_turn",
    }))
}

/// PDF → document block, image → image block, text/code → inline text.
/// Mirrors readFileAsBlock() in ClaudeService.swift, except that a file it cannot
/// inline comes back as the reason rather than as nothing at all.
fn file_block(path: &str) -> Result<Value, String> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let media_type = match ext.as_str() {
        "pdf" => Some(("document", "application/pdf")),
        "jpg" | "jpeg" => Some(("image", "image/jpeg")),
        "png" => Some(("image", "image/png")),
        "gif" => Some(("image", "image/gif")),
        "webp" => Some(("image", "image/webp")),
        _ => None,
    };

    if let Some((block_type, media)) = media_type {
        let bytes = std::fs::read(path).map_err(|e| format!("it could not be read: {e}"))?;
        return Ok(json!({
            "type": block_type,
            "source": { "type": "base64", "media_type": media, "data": base64(&bytes) },
        }));
    }

    let len = std::fs::metadata(path)
        .map_err(|e| format!("it could not be read: {e}"))?
        .len();
    if len > MAX_INLINE_TEXT {
        return Err(format!(
            "it is {} KB and the limit is {} KB",
            len / 1000,
            MAX_INLINE_TEXT / 1000
        ));
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("it could not be read: {e}"))?;
    Ok(json!({ "type": "text", "text": format!("File contents:\n{text}") }))
}

/// Small standalone base64 encoder — not worth another dependency.
/// Also used for Stripe's basic auth.
pub(crate) fn base64_for(bytes: &[u8]) -> String {
    base64(bytes)
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n as usize & 63] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::base64;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }
}
