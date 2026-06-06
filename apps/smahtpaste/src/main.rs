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

#[derive(JsonSchema)]
pub struct SmahtText {
    pub html: String,
    pub markdown: String,
}

fn clipboard_image_data_url(image: ImageData<'_>) -> Result<String, Box<dyn Error>> {
    let expected_len = image
        .width
        .checked_mul(image.height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "clipboard image is too large"))?;

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

fn get_clipboard_content()
-> Result<Option<ChatCompletionRequestUserMessageContentPart>, Box<dyn Error>> {
    let mut clipboard = Clipboard::new()?;

    if let Ok(image) = clipboard.get_image() {
        return Ok(Some(ChatCompletionRequestUserMessageContentPart::ImageUrl(
            ChatCompletionRequestMessageContentPartImage {
                image_url: ImageUrl {
                    url: clipboard_image_data_url(image)?,
                    detail: None,
                },
            },
        )));
    }

    if let Ok(text) = clipboard.get_text() {
        return Ok(Some(ChatCompletionRequestUserMessageContentPart::Text(
            text.into(),
        )));
    }

    Ok(None)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let content = get_clipboard_content()?;
    if content.is_none() {
        return Ok(());
    }

    let content = content.expect("clipboard content should be present");

    let config = OpenAIConfig::new().with_api_base("http:127.0.0.1:14892/v1");

    let client = Client::with_config(config);

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
