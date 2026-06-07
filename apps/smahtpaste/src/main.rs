//! `smahtpaste` — read clipboard content and convert it to HTML & GitHub Flavored Markdown.
//!
//! This binary reads the system clipboard (image or text), sends it to a local
//! OpenAI-compatible endpoint (default `http://127.0.0.1:14892/v1`), and prints
//! the resulting HTML + Markdown JSON to stdout.
//!
//! ## Usage
//!
//! ```bash
//! smahtpaste
//! ```
//!
//! It will detect whether the clipboard holds an image or plain text and
//! accordingly construct a multimodal prompt. The response format is a JSON
//! object with two keys: `html` and `markdown`.

use std::error::Error;
use std::io::{Error as IoError, ErrorKind};

use arboard::{Clipboard, ImageData};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
        ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequestArgs, ImageUrl,
        ResponseFormat, ResponseFormatJsonSchema,
    },
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::ImageEncoder as _;
use schemars::{JsonSchema, schema_for};
use serde_json::json;

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

/// Convert a clipboard [`image::Image`] into a base64-encoded PNG data URL.
///
/// Reads the raw RGBA bytes from the clipboard image, encodes them as PNG,
/// and returns a `data:image/png;base64,...` URL suitable for sending to
/// an OpenAI-compatible multimodal API.
fn clipboard_image_data_url(image: ImageData<'_>) -> Result<String, Box<dyn Error>> {
    // Compute expected byte length: width × height × 4 (RGBA = 4 channels).
    // Use checked arithmetic to avoid overflow on unusually large images.
    let expected_len = image
        .width
        .checked_mul(image.height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "clipboard image is too large"))?;

    // Validate dimensions and byte count before attempting PNG encoding.
    // A mismatch here indicates corrupted or incomplete clipboard data.
    if image.width == 0 || image.height == 0 || image.bytes.len() != expected_len {
        return Err(IoError::new(ErrorKind::InvalidData, "invalid clipboard image data").into());
    }

    let mut png = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut png);
    encoder.write_image(
        image.bytes.as_ref(),
        u32::try_from(image.width)?,
        u32::try_from(image.height)?,
        image::ExtendedColorType::Rgba8,
    )?;

    Ok(format!("data:image/png;base64,{}", STANDARD.encode(png)))
}

/// Read the system clipboard and return its content as a chat message part.
///
/// Prefers image data when available; falls back to plain text.
/// Returns `Ok(None)` if nothing useful is found on the clipboard.
fn get_clipboard_content()
-> Result<Option<ChatCompletionRequestUserMessageContentPart>, Box<dyn Error>> {
    // Initialize a new clipboard handle. This is a blocking operation.
    let mut clipboard = Clipboard::new()?;

    // Try reading image first — images carry richer context for conversion.
    // If the clipboard holds no image, fall through to text.
    if let Ok(image) = clipboard.get_image() {
        return Ok(Some(ChatCompletionRequestUserMessageContentPart::ImageUrl(
            ChatCompletionRequestMessageContentPartImage {
                image_url: ImageUrl {
                    // Encode the raw RGBA bytes as a PNG data URL for the API.
                    url: clipboard_image_data_url(image)?,
                    detail: None, // default quality ("auto")
                },
            },
        )));
    }

    // Fall back to plain text if no image was available.
    if let Ok(text) = clipboard.get_text() {
        return Ok(Some(ChatCompletionRequestUserMessageContentPart::Text(
            text.into(),
        )));
    }

    // Nothing readable was found — no-op (the caller will exit early).
    Ok(None)
}

/// Entry point: read clipboard, send to local LLM, print JSON result.
///
/// 1. Reads clipboard content (image → text fallback) via [`get_clipboard_content`].
/// 2. Sends the content to a local OpenAI-compatible endpoint (default
///    `http://127.0.0.1:14892/v1`) using the `Qwen3.5-9B-Heretic-mxfp4` model.
/// 3. Expects a JSON response matching [`SmahtText`] schema.
/// 4. Prints each choice's content to stdout.
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Read clipboard; if nothing was found, exit silently.
    let content = get_clipboard_content()?;
    if content.is_none() {
        return Ok(());
    }

    // Safety: we know content is Some because we just checked is_none above.
    let content = content.expect("clipboard content should be present");

    // Connect to a local OpenAI-compatible endpoint.
    // NOTE: this URL may need updating if the local server port changes.
    let config = OpenAIConfig::new().with_api_base("http://127.0.0.1:14892/v1");

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
        .model("Qwen3.5-9B-Heretic-mxfp4")
        .messages([
            ChatCompletionRequestSystemMessage::from(
                "Create HTML and Markdown representation of the content it may be image contents to text formats or text transformations to other text formats. The HTML should be minimal, just what would be inside of the body excluding the body, no styling, only required attributes, core elements.",
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
