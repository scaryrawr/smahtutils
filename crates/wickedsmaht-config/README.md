# wickedsmaht-config

`wickedsmaht-config` is the shared configuration crate for the workspace. It loads optional defaults from `$HOME/.wickedsmaht/config.json` and provides `ResolvableSetting` so apps can prefer CLI values over configured values.

This crate supports vibe-coded personal dev utilities, not polished products. Your mileage may vary.

## Config file

The default config path is:

```text
$HOME/.wickedsmaht/config.json
```

Supported keys are optional:

```json
{
  "base_url": "http://127.0.0.1:14892/v1",
  "model": "my-chat-model",
  "coding_embedding_model": "my-embedding-model"
}
```

Kebab-case aliases are also accepted:

```json
{
  "base-url": "http://127.0.0.1:14892/v1",
  "coding-embedding-model": "my-embedding-model"
}
```

## API

`Config::load()` reads the default path. A missing file returns `Config::default()`, while unreadable files and invalid JSON are returned as errors.

`Config::load_from_path(path)` is available for tests and callers that need an explicit path.

`ResolvableSetting` is implemented for `String` and resolves values in this order:

1. CLI value.
2. Config value.
3. Error naming the missing flag and config key.

## Development

```bash
cargo test -p wickedsmaht-config
cargo test -p wickedsmaht-config setting_resolution_tests::
```

Keep shared defaults and config-resolution behavior here instead of duplicating it in apps.
