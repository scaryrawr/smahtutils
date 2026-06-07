//! Clipboard reading utilities for wickedpaste.
//!
//! This module handles reading content from the system clipboard,
//! supporting both image and text formats.

use std::error::Error;
use std::io::{Error as IoError, ErrorKind};

use arboard::{Clipboard, ImageData};
use async_openai::types::chat::{
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestUserMessageContentPart,
    ImageUrl,
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::ImageEncoder as _;

/// Convert a clipboard [`image::Image`] into a base64-encoded PNG data URL.
///
/// Reads the raw RGBA bytes from the clipboard image, encodes them as PNG,
/// and returns a `data:image/png;base64,...` URL suitable for sending to
/// an OpenAI-compatible multimodal API.
pub fn clipboard_image_data_url(image: ImageData<'_>) -> Result<String, Box<dyn Error>> {
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
pub fn get_clipboard_content()
-> Result<Option<ChatCompletionRequestUserMessageContentPart>, Box<dyn Error>> {
    // Initialize a new clipboard handle. This is a blocking operation.
    let mut clipboard = Clipboard::new()?;

    // Try reading image first — images carry richer context for conversion.
    // If the clipboard holds no image, fall through to text.
    if let Ok(image) = clipboard.get_image() {
        return Ok(Some(
            ChatCompletionRequestUserMessageContentPart::ImageUrl(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageUrl {
                        // Encode the raw RGBA bytes as a PNG data URL for the API.
                        url: clipboard_image_data_url(image)?,
                        detail: None, // default quality ("auto")
                    },
                },
            ),
        ));
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
