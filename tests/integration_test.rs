use canary_rs::{Canary, ExecutionConfig, Result, SessionConfig};

#[test]
#[ignore]
fn test_transcribe_loading_audio() -> Result<()> {
    let model_path = "canary-1b-v2";
    let model = Canary::from_pretrained(model_path, None)?;
    let mut session = model.session();

    // Test with the loading.raw file
    let audio_path = "audio.wav";
    let result = session.transcribe_file(audio_path, "en", "en")?;

    println!("Transcription result: {}", result.text);
    println!("Tokens: {:#?}", result.tokens);

    // The audio should say something like "Multitask loaded successfully"
    // We'll just check it's not empty for now
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
#[ignore]
fn test_vocab_has_canary2_tokens_1b() {
    let vocab = std::fs::read_to_string("canary-1b-v2/vocab.txt")
        .expect("missing vocab.txt for canary-1b-v2");
    assert!(vocab.contains("<|startofcontext|>"));
    assert!(vocab.contains("<|emo:undefined|>"));
}

#[test]
#[ignore]
fn test_vocab_has_canary2_tokens_180m() {
    let vocab = std::fs::read_to_string("canary-180m-flash/vocab.txt")
        .expect("missing vocab.txt for canary-180m-flash");
    assert!(vocab.contains("<|startofcontext|>"));
    assert!(vocab.contains("<|emo:undefined|>"));
}

#[test]
#[ignore]
fn test_model_loads() -> Result<()> {
    let model_path = "canary-1b-v2";
    let _model = Canary::from_pretrained(model_path, None)?;
    println!("Model loaded successfully!");
    Ok(())
}
