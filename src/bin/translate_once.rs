use canary_rs::{Canary, ExecutionConfig, ExecutionProvider, SessionConfig};

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|v| v.trim().parse::<usize>().ok())
}

fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key).ok().and_then(|v| v.trim().parse::<f32>().ok())
}

fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key)
        .ok()
        .and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn env_string(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn usage(bin: &str) {
    eprintln!(
        "Usage: {bin} <audio_path> <source_lang> <target_lang> [model_dir]\n\n\
         Examples:\n\
           {bin} /path/to/audio.wav en de\n\
           {bin} /path/to/audio.wav de en canary-1b-v2\n\n\
         Env:\n\
           CANARY_MODEL_DIR             Default model dir if [model_dir] arg is omitted\n\
           CANARY_EXECUTION_PROVIDER    cpu (default) | cuda | tensorrt | coreml | directml | rocm | openvino | webgpu | nnapi\n\
           CANARY_CUDA_DEVICE_ID        CUDA/TensorRT device index (default: 0)\n\
           CANARY_ITN                   1/true => <|itn|>, 0/false => <|noitn|>, unset => omit\n\
           CANARY_BEAM_SIZE             integer, default library value\n\
           CANARY_MAX_LENGTH            integer, default library value\n\
           CANARY_REPETITION_PENALTY    float, default library value\n\
           CANARY_LENGTH_PENALTY        float, default library value\n\
           CANARY_SAMPLE                1/0 to enable sampling\n\
           CANARY_TEMPERATURE           float\n\
           CANARY_TOP_K                 integer\n\
           CANARY_TOP_P                 float\n\
           CANARY_PROMPT_OVERRIDE       explicit prompt string\n\
           CANARY_DECODER_CONTEXT       decoder context string\n\
           CANARY_EMOTION_TOKEN         Canary2 emotion token\n\
           CANARY_CHUNK_SECONDS         long-audio chunk size in seconds (e.g. 15)"
    );
}

fn parse_provider() -> ExecutionProvider {
    match std::env::var("CANARY_EXECUTION_PROVIDER")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "cuda" => ExecutionProvider::Cuda,
        "tensorrt" => ExecutionProvider::TensorRT,
        "coreml" => ExecutionProvider::CoreML,
        "directml" => ExecutionProvider::DirectML,
        "rocm" => ExecutionProvider::ROCm,
        "openvino" => ExecutionProvider::OpenVINO,
        "webgpu" => ExecutionProvider::WebGPU,
        "nnapi" => ExecutionProvider::NNAPI,
        _ => ExecutionProvider::Cpu,
    }
}

fn parse_itn() -> Option<bool> {
    std::env::var("CANARY_ITN").ok().and_then(|v| {
        match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" => Some(true),
            "0" | "false" => Some(false),
            _ => None,
        }
    })
}



fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let bin = args.next().unwrap_or_else(|| "translate_once".to_string());

    let audio_path = match args.next() {
        Some(v) => v,
        None => {
            usage(&bin);
            std::process::exit(2);
        }
    };
    let source_lang = match args.next() {
        Some(v) => v,
        None => {
            usage(&bin);
            std::process::exit(2);
        }
    };
    let target_lang = match args.next() {
        Some(v) => v,
        None => {
            usage(&bin);
            std::process::exit(2);
        }
    };

    let model_dir = args
        .next()
        .or_else(|| std::env::var("CANARY_MODEL_DIR").ok())
        .unwrap_or_else(|| "canary-180m-flash".to_string());

    let provider = parse_provider();
    let mut config = ExecutionConfig::new().with_execution_provider(provider.clone());

    if matches!(provider, ExecutionProvider::Cuda | ExecutionProvider::TensorRT) {
        let device_id = std::env::var("CANARY_CUDA_DEVICE_ID")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0);
        config = config.with_cuda_device_id(device_id);
    }

    let mut session_cfg = SessionConfig::new().with_itn(parse_itn());

    if let Some(v) = env_usize("CANARY_BEAM_SIZE") {
        session_cfg = session_cfg.with_beam_size(v.max(1));
    }
    if let Some(v) = env_usize("CANARY_MAX_LENGTH") {
        session_cfg = session_cfg.with_max_length(v.max(1));
    }
    if let Some(v) = env_f32("CANARY_REPETITION_PENALTY") {
        session_cfg = session_cfg.with_repetition_penalty(v.max(0.1));
    }
    if let Some(v) = env_f32("CANARY_LENGTH_PENALTY") {
        session_cfg = session_cfg.with_length_penalty(v.max(0.0));
    }
    if let Some(v) = env_bool("CANARY_SAMPLE") {
        session_cfg = session_cfg.with_sampling(v);
    }
    if let Some(v) = env_f32("CANARY_TEMPERATURE") {
        session_cfg = session_cfg.with_temperature(v.max(0.01));
    }
    if let Some(v) = env_usize("CANARY_TOP_K") {
        session_cfg = session_cfg.with_top_k(v);
    }
    if let Some(v) = env_f32("CANARY_TOP_P") {
        session_cfg = session_cfg.with_top_p(v.clamp(0.0, 1.0));
    }
    if let Some(v) = env_string("CANARY_EMOTION_TOKEN") {
        session_cfg = session_cfg.with_emotion_token(v);
    }
    if let Some(v) = env_string("CANARY_DECODER_CONTEXT") {
        session_cfg = session_cfg.with_decoder_context(v);
    }
    if let Some(v) = env_string("CANARY_PROMPT_OVERRIDE") {
        session_cfg.prompt_override = Some(v);
    }
    if let Some(v) = env_usize("CANARY_CHUNK_SECONDS") {
        session_cfg = session_cfg.with_chunk_seconds(v);
    }

    config = config.with_session_config(session_cfg);

    eprintln!("model_dir={model_dir}");
    eprintln!("provider={provider:?}");
    eprintln!("source_lang={source_lang} target_lang={target_lang}");

    let model = Canary::from_pretrained(&model_dir, Some(config))?;
    let mut session = model.session();
    let result = session.transcribe_file(&audio_path, &source_lang, &target_lang)?;

    println!("{}", result.text.trim());
    Ok(())
}
