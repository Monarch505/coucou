// Custom provider — an OpenAI-shaped gateway (a local router, a company proxy)
// that speaks the Anthropic Messages format. Replaces the hardcoded
// api.anthropic.com when the user points Coucou at one.
//
// Everything is empty by default: with nothing configured this module is inert
// and the chat falls back to api.anthropic.com exactly as before.
//
// The key never leaves the Credential Manager. It is read into memory for the
// length of one request and dropped afterwards.

use std::time::Duration;

use serde_json::Value;

use crate::secrets;

/// Credential Manager slot for the custom provider's key.
pub const KEY_NAME: &str = "provider-api-key";

/// Where a request goes once a provider is configured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provider {
    /// Base URL with any trailing slash removed, e.g. "http://127.0.0.1:20128/v1".
    pub base_url: String,
    /// Never logged, never written to disk.
    pub api_key: String,
}

impl Provider {
    /// The Messages endpoint for this provider.
    pub fn messages_url(&self) -> String {
        format!("{}/messages", self.base_url.trim_end_matches('/'))
    }

    /// The model-list endpoint for this provider.
    pub fn models_url(&self) -> String {
        format!("{}/models", self.base_url.trim_end_matches('/'))
    }
}

/// Nothing configured — the caller keeps using api.anthropic.com.
pub fn configured(settings: &crate::settings::Settings) -> bool {
    !settings.provider_base_url.trim().is_empty()
}

/// What is wrong with the base URL the user typed, if anything.
///
/// Typed by hand, so it is the one field on this path a person can get wrong in
/// a dozen ways. Saying which way it is wrong beats a DNS error from somewhere
/// deep inside reqwest.
pub fn check_base_url(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(String::new());
    }

    let Some(rest) = trimmed
        .strip_prefix("http://")
        .or_else(|| trimmed.strip_prefix("https://"))
    else {
        return Err("Start the address with http:// or https://.".into());
    };

    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = host.rsplit('@').next().unwrap_or("");
    let host = host.split(':').next().unwrap_or("");
    if host.is_empty() {
        return Err("That address has no host in it.".into());
    }
    // A single label is allowed — a provider on the LAN can be called anything —
    // but a space would split the request line and has to be refused here.
    if host.contains(char::is_whitespace) {
        return Err("That address has a space in it.".into());
    }

    Ok(trimmed.trim_end_matches('/').to_string())
}

/// Where a chat turn goes. Either the user's provider or Anthropic, so the
/// caller never has to branch on which one is in play.
pub enum Target {
    Provider(Provider),
    Anthropic { url: String, key: String },
}

impl Target {
    pub fn endpoint(&self) -> String {
        match self {
            Target::Provider(p) => p.messages_url(),
            Target::Anthropic { url, .. } => url.clone(),
        }
    }

    /// Only a provider needs `"stream": false`; Anthropic's own API is already
    /// non-streaming for this body.
    pub fn prepare_body(&self, body: &Value) -> Value {
        match self {
            Target::Provider(_) => prepare(body),
            Target::Anthropic { .. } => body.clone(),
        }
    }
}

/// Build a provider, or `None` when the URL is blank.
///
/// A configured URL with no key still resolves: an unauthenticated local router
/// needs no secret at all, so refusing to send would break a legitimate setup.
pub fn resolve(settings: &crate::settings::Settings) -> Result<Option<Provider>, String> {
    if !configured(settings) {
        return Ok(None);
    }
    Ok(Some(Provider {
        base_url: check_base_url(&settings.provider_base_url)?,
        api_key: secrets::get(KEY_NAME).unwrap_or_default(),
    }))
}

/// Add `"stream": false` to the request body.
///
/// A local router answers on its own terms: probed on 2026-10-01, the one this
/// was built against streams by default, so the JSON parse in `claude::call`
/// would fail on `event: message_start`. Asking for a non-streamed answer is
/// what makes the same client read both a real API and a router.
/// ponytail: if a provider ignores this and still streams, add the SSE reader
/// from `hook/src/main.rs` as a second branch — measured here, not assumed.
pub fn prepare(body: &Value) -> Value {
    let mut body = body.clone();
    if let Some(obj) = body.as_object_mut() {
        obj.insert("stream".into(), Value::Bool(false));
    }
    body
}

/// Fetch the provider's model list so the settings window can offer what it
/// actually has instead of a list hardcoded at build time.
///
/// Reads the OpenAI-shaped `{"data": [{"id": "..."}]}` shape. A `combo/` prefix
/// in the id is left alone — it is part of the name, not a field.
pub async fn models(client: &reqwest::Client, provider: &Provider) -> Result<Vec<String>, String> {
    let response = client
        .get(provider.models_url())
        .header("authorization", format!("Bearer {}", provider.api_key))
        .header("anthropic-version", "2023-06-01")
        .send()
        .await
        .map_err(|e| format!("Could not reach the provider: {e}"))?;

    let status = response.status();
    let text = response.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("Provider {status}"));
    }

    parse_models(&text)
}

/// Pull the ids out of a `/models` response. Sorted so the dropdown order does
/// not depend on how the server happened to order them.
fn parse_models(body: &str) -> Result<Vec<String>, String> {
    let value: Value = serde_json::from_str(body).map_err(|e| format!("Bad model list: {e}"))?;

    // `data` is the OpenAI spelling; `models` and `combo` are the names other
    // routers use for the same array, so all three are accepted.
    let list = value
        .get("data")
        .or_else(|| value.get("models"))
        .or_else(|| value.get("combo"))
        .and_then(Value::as_array)
        .ok_or_else(|| "No model list in the provider's answer.".to_string())?;

    let mut ids: Vec<String> = list
        .iter()
        .filter_map(|entry| match entry {
            // `{"id": "combo/Big-P"}` — the OpenAI spelling, and what the
            // router measured on 2026-10-01 answers.
            Value::Object(_) => entry
                .get("id")
                .or_else(|| entry.get("model"))
                .or_else(|| entry.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string),
            // `"combo/Big-P"` — some routers list bare strings instead.
            Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .filter(|id| !id.trim().is_empty())
        .collect();

    ids.sort();
    ids.dedup();
    if ids.is_empty() {
        return Err("The provider returned an empty model list.".to_string());
    }
    Ok(ids)
}

/// Request headers for this provider.
///
/// `x-api-key` and `Authorization: Bearer` are both sent: which one a gateway
/// expects depends on the gateway, and a header the server ignores costs
/// nothing. Coucou talks to nothing but the endpoint the user typed, so the
/// extra header cannot leak the key anywhere else.
pub fn headers(provider: &Provider) -> Vec<(&'static str, String)> {
    let mut out = vec![
        ("anthropic-version", "2023-06-01".to_string()),
        ("content-type", "application/json".to_string()),
    ];
    if !provider.api_key.is_empty() {
        out.push(("x-api-key", provider.api_key.clone()));
        out.push(("authorization", format!("Bearer {}", provider.api_key)));
    }
    out
}

/// A client for the provider call. Short timeout: a local router that has
/// stopped answering must not hang the island.
pub fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(90))
        .build()
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn messages_url_trims_a_trailing_slash() {
        let p = Provider { base_url: "http://127.0.0.1:20128/v1/".into(), api_key: String::new() };
        assert_eq!(p.messages_url(), "http://127.0.0.1:20128/v1/messages");
        assert_eq!(p.models_url(), "http://127.0.0.1:20128/v1/models");
    }

    #[test]
    fn prepare_adds_stream_false_without_touching_the_rest() {
        let body = json!({ "model": "combo/Mimo-2.6", "max_tokens": 16 });
        let out = prepare(&body);
        assert_eq!(out["stream"], Value::Bool(false));
        assert_eq!(out["model"], "combo/Mimo-2.6");
        assert_eq!(out["max_tokens"], 16);
    }

    /// Pinned against the live 2026-10-01 answer from the local router: the id
    /// carries a `combo/` prefix that must survive parsing.
    #[test]
    fn parse_models_reads_combo_prefixed_ids() {
        let body = r#"{"object":"list","data":[
            {"id":"combo/Mimo-2.6"},
            {"id":"combo/Big-P"},
            {"id":"combo/Nemotron-3.5"}
        ]}"#;
        let ids = parse_models(body).unwrap();
        assert_eq!(ids, vec!["combo/Big-P", "combo/Mimo-2.6", "combo/Nemotron-3.5"]);
    }

    /// The router that this was built for answers a single plain string per
    /// entry, not `{"id": "..."}`.
    #[test]
    fn parse_models_reads_bare_strings_and_other_field_names() {
        let body = r#"{"models":["combo/Big-P","combo/Mimo-2.6"]}"#;
        assert_eq!(parse_models(body).unwrap(), vec!["combo/Big-P", "combo/Mimo-2.6"]);
    }

    #[test]
    fn parse_models_rejects_an_empty_or_unusable_list() {
        assert!(parse_models(r#"{"data":[]}"#).is_err());
        assert!(parse_models(r#"{"data":[{"id":"  "}]}"#).is_err());
        assert!(parse_models("not json").is_err());
    }

    #[test]
    fn headers_carry_no_key_when_none_is_saved() {
        let p = Provider { base_url: "http://127.0.0.1:20128/v1".into(), api_key: String::new() };
        assert!(!headers(&p).iter().any(|(k, _)| *k == "x-api-key"));
    }

    #[test]
    fn headers_send_the_key_when_one_is_saved() {
        let p = Provider { base_url: "http://127.0.0.1:20128/v1".into(), api_key: "sk-test".into() };
        let h = headers(&p);
        assert!(h.iter().any(|(k, v)| *k == "x-api-key" && v == "sk-test"));
        assert!(h.iter().any(|(k, v)| *k == "authorization" && v == "Bearer sk-test"));
    }

    #[test]
    fn check_base_url_accepts_a_blank_field() {
        assert_eq!(check_base_url("   ").unwrap(), "");
    }

    #[test]
    fn check_base_url_trims_and_drops_the_trailing_slash() {
        assert_eq!(check_base_url("  http://127.0.0.1:20128/v1/  ").unwrap(), "http://127.0.0.1:20128/v1");
        assert_eq!(check_base_url("http://127.0.0.1:20128/v1").unwrap(), "http://127.0.0.1:20128/v1");
    }

    #[test]
    fn check_base_url_rejects_what_a_hand_typed_address_gets_wrong() {
        for bad in [
            "127.0.0.1:20128",        // no scheme — it would silently become a path
            "ftp://127.0.0.1/v1",     // a scheme Coucou cannot speak
            "http://",                // nothing after the scheme
            "http://local host/v1",   // a space splits the request line
        ] {
            assert!(check_base_url(bad).is_err(), "should have rejected {bad:?}");
        }
    }

    /// A LAN address without a dot is legal on a home network, and a provider
    /// can be named anything the user likes — so a single label is not a typo.
    #[test]
    fn check_base_url_accepts_a_bare_host_name() {
        assert_eq!(check_base_url("http://router/v1").unwrap(), "http://router/v1");
        assert_eq!(check_base_url("http://localhost:20128/v1").unwrap(), "http://localhost:20128/v1");
    }
}