use std::{io, net::AddrParseError};

use thiserror::Error;
use wickedsmaht_config::{ConfigError, SettingError};

#[derive(Debug, Error)]
pub enum SmahtiesError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Setting(#[from] SettingError),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Sql(#[from] rusqlite::Error),
    #[error(transparent)]
    OpenAi(#[from] async_openai::error::OpenAIError),
    #[error(transparent)]
    Notify(#[from] notify::Error),
    #[error(transparent)]
    TreeSitterLanguage(#[from] tree_sitter::LanguageError),
    #[error(transparent)]
    AddrParse(#[from] AddrParseError),
    #[error("{0}")]
    InvalidRequest(String),
}

pub type Result<T> = std::result::Result<T, SmahtiesError>;
