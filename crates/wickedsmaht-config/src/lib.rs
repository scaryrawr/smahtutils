//! Shared configuration loading for wickedsmaht applications.
//!
//! Configuration is read from `$HOME/.wickedsmaht/config.json` when an
//! application needs defaults. The current shared keys are:
//!
//! ```json
//! {
//!   "base_url": "http://127.0.0.1:14892/v1",
//!   "model": "my-model"
//! }
//! ```
//!
//! `base-url` is also accepted as an alias for `base_url` when deserializing.

use std::{
    env,
    error::Error,
    fmt, fs, io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

/// Directory name under `$HOME` that stores wickedsmaht configuration.
pub const CONFIG_DIR_NAME: &str = ".wickedsmaht";
/// Configuration file name inside [`CONFIG_DIR_NAME`].
pub const CONFIG_FILE_NAME: &str = "config.json";

/// Shared wickedsmaht configuration.
///
/// All fields are optional so applications can decide which settings are
/// required and can let command line arguments override configured defaults.
#[derive(Debug, Clone, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct Config {
    /// Default OpenAI-compatible API base URL.
    #[serde(alias = "base-url")]
    pub base_url: Option<String>,
    /// Default model name.
    pub model: Option<String>,
}

impl Config {
    /// Load configuration from `$HOME/.wickedsmaht/config.json`.
    ///
    /// A missing config file is treated as an empty/default config. Other read
    /// errors and JSON parse errors are returned to the caller.
    pub fn load() -> Result<Self, ConfigError> {
        Self::load_from_path(default_config_path()?)
    }

    /// Load configuration from a specific path.
    ///
    /// This is useful for tests and for applications that want to support an
    /// explicit config path later.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        match fs::read_to_string(path) {
            Ok(contents) => serde_json::from_str(&contents).map_err(|source| ConfigError::Parse {
                path: path.to_path_buf(),
                source,
            }),
            Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(ConfigError::Read {
                path: path.to_path_buf(),
                source,
            }),
        }
    }
}

/// Return the default configuration path, `$HOME/.wickedsmaht/config.json`.
pub fn default_config_path() -> Result<PathBuf, ConfigError> {
    let home = env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .ok_or(ConfigError::HomeNotFound)?;

    Ok(config_path_from_home(home))
}

/// Build the wickedsmaht configuration path for a given home directory.
pub fn config_path_from_home(home: impl Into<PathBuf>) -> PathBuf {
    home.into().join(CONFIG_DIR_NAME).join(CONFIG_FILE_NAME)
}

/// Errors that can occur while locating or loading wickedsmaht configuration.
#[derive(Debug)]
pub enum ConfigError {
    /// `$HOME` is not set, so the default configuration path cannot be found.
    HomeNotFound,
    /// The configuration file could not be read.
    Read { path: PathBuf, source: io::Error },
    /// The configuration file was read but did not contain valid JSON config.
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HomeNotFound => write!(
                f,
                "could not locate wickedsmaht config: HOME environment variable is not set"
            ),
            Self::Read { path, source } => {
                write!(f, "failed to read config {}: {source}", path.display())
            }
            Self::Parse { path, source } => {
                write!(f, "failed to parse config {}: {source}", path.display())
            }
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::HomeNotFound => None,
            Self::Read { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn config_path_uses_wickedsmaht_directory() {
        assert_eq!(
            config_path_from_home("/home/example"),
            PathBuf::from("/home/example/.wickedsmaht/config.json")
        );
    }

    #[test]
    fn missing_config_file_loads_default_config() {
        let path = unique_test_path("missing-config.json");

        assert_eq!(Config::load_from_path(path).unwrap(), Config::default());
    }

    #[test]
    fn parses_base_url_and_model() {
        let path = unique_test_path("config.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            r#"{
                "base_url": "http://127.0.0.1:14892/v1",
                "model": "local-model"
            }"#,
        )
        .unwrap();

        assert_eq!(
            Config::load_from_path(&path).unwrap(),
            Config {
                base_url: Some("http://127.0.0.1:14892/v1".into()),
                model: Some("local-model".into()),
            }
        );
    }

    #[test]
    fn parses_base_url_alias() {
        let path = unique_test_path("alias-config.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            r#"{
                "base-url": "http://127.0.0.1:14892/v1",
                "model": "local-model"
            }"#,
        )
        .unwrap();

        assert_eq!(
            Config::load_from_path(&path).unwrap().base_url,
            Some("http://127.0.0.1:14892/v1".into())
        );
    }

    fn unique_test_path(file_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        env::temp_dir().join(format!(
            "wickedsmaht-config-test-{}-{nanos}/{file_name}",
            std::process::id()
        ))
    }
}
