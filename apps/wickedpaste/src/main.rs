//! `wickedpaste` — read clipboard content and convert it to HTML & GitHub Flavored Markdown.
//!
//! This binary reads the system clipboard (image or text), sends it to an
//! OpenAI-compatible endpoint, and prints the resulting HTML + Markdown JSON to
//! stdout.
//!
//! ## Usage
//!
//! ```bash
//! wickedpaste --base-url http://127.0.0.1:14892/v1 --model my-model
//! ```
//!
//! It will detect whether the clipboard holds an image or plain text and
//! accordingly construct a multimodal prompt. The response format is a JSON
//! object with two keys: `html` and `markdown`.

use std::error::Error;

mod clipboard;

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
        CreateChatCompletionRequestArgs, ResponseFormat, ResponseFormatJsonSchema,
    },
};
use clap::Parser;
use schemars::{JsonSchema, schema_for};
use serde_json::json;
use wickedsmaht_config::{Config, ResolvableSetting};

use crate::clipboard::get_clipboard_content;

/// The expected shape of the LLM's JSON response.
///
/// Contains both HTML and GitHub Flavored Markdown representations
/// of the clipboard content.
#[derive(JsonSchema)]
pub struct SmahtText {
    /// Minimal HTML (body contents only, no wrapper elements).
    pub html: String,
    /// GitHub Flavored Markdown equivalent.
    pub markdown: String,
}

/// Command line arguments for wickedpaste.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The base URL of the OpenAI-compatible API endpoint.
    ///
    /// If omitted, wickedpaste reads `base_url` from
    /// `$HOME/.wickedsmaht/config.json`.
    #[arg(long)]
    base_url: Option<String>,

    /// The model name to use.
    ///
    /// If omitted, wickedpaste reads `model` from
    /// `$HOME/.wickedsmaht/config.json`.
    #[arg(long)]
    model: Option<String>,
}

/// Entry point: read clipboard, send to local LLM, print JSON result.
///
/// 1. Reads clipboard content (image → text fallback) via [`get_clipboard_content`].
/// 2. Sends the content to the specified OpenAI-compatible endpoint using the specified model.
/// 3. Expects a JSON response matching [`SmahtText`] schema.
/// 4. Prints each choice's content to stdout.
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let (base_url, model) = resolve_api_settings(args)?;

    // Read clipboard; if nothing was found, exit silently.
    let Some(content) = get_clipboard_content()? else {
        return Ok(());
    };

    // Connect to the specified OpenAI-compatible endpoint.
    let config = OpenAIConfig::new().with_api_base(&base_url);

    let client = Client::with_config(config);

    // Build a JSON schema from [`SmahtText`] so the LLM returns structured output.
    let schema = json!(schema_for!(SmahtText));

    let response_format = ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: None,
            name: "smaht_text".into(),
            schema,
            strict: Some(true),
        },
    };

    let user_message = ChatCompletionRequestUserMessage::from(
        ChatCompletionRequestUserMessageContent::Array(vec![
            ChatCompletionRequestUserMessageContentPart::Text(
                "Convert the following into HTML and GitHub Flavored Markdown".into(),
            ),
            content,
        ]),
    );

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(512u32)
        .model(&model)
        .messages([
            ChatCompletionRequestSystemMessage::from(
                "Create HTML and Markdown representation of the content it may be image contents to text formats or text transformations to other text formats. The HTML should be minimal, we don't need the <html> or other top level tags, no body, no styling, avoid adding unneeded whitespace to HTML, very plain minimal unfancy html elements.",
            )
            .into(),
            user_message.into(),
        ])
        .response_format(response_format)
        .build()?;

    let response = client.chat().create(request).await?;

    for choice in response.choices {
        if let Some(content) = choice.message.content {
            print!("{content}")
        }
    }

    Ok(())
}

fn resolve_api_settings(args: Args) -> Result<(String, String), Box<dyn Error>> {
    let config = if args.base_url.is_some() && args.model.is_some() {
        Config::default()
    } else {
        Config::load()?
    };

    let base_url = String::resolve(args.base_url, config.base_url, "--base-url", "base_url")?;
    let model = String::resolve(args.model, config.model, "--model", "model")?;

    Ok((base_url, model))
}

#[cfg(test)]
mod tests {}
