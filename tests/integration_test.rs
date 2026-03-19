use canary_rs::{Canary, ExecutionConfig, Result, SessionConfig};
use std::path::Path;

fn require_path(key: &str, default: &str) -> Option<String> {
    let value = std::env::var(key).unwrap_or_else(|_| default.to_string());
    if Path::new(&value).exists() {
        Some(value)
    } else {
        None
    }
}

#[test]
fn test_transcribe_loading_audio() -> Result<()> {
    let model_path = match require_path("CANARY_TEST_MODEL_DIR", "canary-1b-v2") {
        Some(p) => p,
        None => {
            eprintln!("Skipping: missing CANARY_TEST_MODEL_DIR");
            return Ok(());
        }
    };

    let audio_path = match require_path("CANARY_TEST_AUDIO", "audio.wav") {
        Some(p) => p,
        None => {
            eprintln!("Skipping: missing CANARY_TEST_AUDIO");
            return Ok(());
        }
    };

    let model = Canary::from_pretrained(model_path, None)?;
    let mut session = model.session();

    let result = session.transcribe_file(audio_path, "en", "en")?;

    println!("Transcription result: {}", result.text);
    println!("Tokens: {:#?}", result.tokens);

    assert!(!result.text.is_empty(), "Transcription should not be empty");

    Ok(())
}

#[test]
fn test_session_config_defaults() {
    let cfg = SessionConfig::default();
    assert!(cfg.use_pnc);
    assert!(!cfg.use_itn);
    assert!(!cfg.use_timestamps);
    assert!(!cfg.use_diarize);
    assert_eq!(cfg.max_length, 512);
    assert_eq!(cfg.beam_size, 1);
    assert_eq!(cfg.length_penalty, 1.0);
    assert_eq!(cfg.repetition_penalty, 1.0);
    assert!(cfg.suppress_tokens_below.is_infinite());
    assert!(cfg.suppress_token_ids.is_none());
    assert!(cfg.emotion_token.is_none());
    assert!(!cfg.sample);
    assert_eq!(cfg.temperature, 1.0);
    assert_eq!(cfg.top_k, 0);
    assert_eq!(cfg.top_p, 1.0);
}

#[test]
fn test_session_config_builders() {
    let cfg = SessionConfig::new()
        .with_max_length(256)
        .with_beam_size(4)
        .with_length_penalty(0.9)
        .with_repetition_penalty(1.2)
        .with_suppress_tokens(vec![1, 2, 3])
        .with_suppress_tokens_below(-5.0)
        .with_emotion_token("<|emo:neutral|>")
        .with_sampling(true)
        .with_temperature(0.8)
        .with_top_k(40)
        .with_top_p(0.9);

    assert_eq!(cfg.max_length, 256);
    assert_eq!(cfg.beam_size, 4);
    assert_eq!(cfg.length_penalty, 0.9);
    assert_eq!(cfg.repetition_penalty, 1.2);
    assert_eq!(cfg.suppress_tokens_below, -5.0);
    assert_eq!(cfg.suppress_token_ids, Some(vec![1, 2, 3]));
    assert_eq!(cfg.emotion_token.as_deref(), Some("<|emo:neutral|>"));
    assert!(cfg.sample);
    assert_eq!(cfg.temperature, 0.8);
    assert_eq!(cfg.top_k, 40);
    assert_eq!(cfg.top_p, 0.9);
}

#[test]
fn test_execution_config_uses_session_config() {
    let session_cfg = SessionConfig::new().with_beam_size(3).with_sampling(true);
    let exec_cfg = ExecutionConfig::new().with_session_config(session_cfg.clone());
    assert_eq!(exec_cfg.session.beam_size, session_cfg.beam_size);
    assert_eq!(exec_cfg.session.sample, session_cfg.sample);
}

#[test]
fn test_vocab_has_canary2_tokens_1b() {
    let model_dir = match require_path("CANARY_TEST_MODEL_DIR", "canary-1b-v2") {
        Some(p) => p,
        None => {
            eprintln!("Skipping: missing CANARY_TEST_MODEL_DIR");
            return;
        }
    };
    let vocab_path = format!("{}/vocab.txt", model_dir);
    if !Path::new(&vocab_path).exists() {
        eprintln!("Skipping: missing vocab.txt at {}", vocab_path);
        return;
    }
    let vocab = std::fs::read_to_string(vocab_path).expect("missing vocab.txt");
    assert!(vocab.contains("<|startofcontext|>"));
    assert!(vocab.contains("<|emo:undefined|>"));
}

#[test]
fn test_vocab_has_canary2_tokens_180m() {
    let model_dir = match require_path("CANARY_TEST_MODEL_DIR_180M", "canary-180m-flash") {
        Some(p) => p,
        None => {
            eprintln!("Skipping: missing CANARY_TEST_MODEL_DIR_180M");
            return;
        }
    };
    let vocab_path = format!("{}/vocab.txt", model_dir);
    if !Path::new(&vocab_path).exists() {
        eprintln!("Skipping: missing vocab.txt at {}", vocab_path);
        return;
    }
    let vocab = std::fs::read_to_string(vocab_path).expect("missing vocab.txt");
    assert!(vocab.contains("<|startofcontext|>"));
    assert!(vocab.contains("<|emo:undefined|>"));
}

#[test]
fn test_model_loads() -> Result<()> {
    let model_path = match require_path("CANARY_TEST_MODEL_DIR", "canary-1b-v2") {
        Some(p) => p,
        None => {
            eprintln!("Skipping: missing CANARY_TEST_MODEL_DIR");
            return Ok(());
        }
    };
    let _model = Canary::from_pretrained(model_path, None)?;
    println!("Model loaded successfully!");
    Ok(())
}
