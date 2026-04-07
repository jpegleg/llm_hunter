use chrono::Utc;
use std::any::Any;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
use std::fs::File;
use std::io::{self, Read};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;

const ENTROPY_SAMPLE_MAX: usize = 4096;
const TEXT_SAMPLE_MAX: usize = 8192;
const DEFAULT_SCAN_WINDOW: usize = 512 * 1024;
const DEFAULT_DEEP_SCAN_CHUNK_BYTES: usize = 1024 * 1024;
const DEFAULT_DEEP_SCAN_OVERLAP_BYTES: usize = 16 * 1024;
const DEFAULT_DEEP_ENTROPY_WINDOW_BYTES: usize = 16 * 1024;
const DEFAULT_MAX_REPORTED_ANOMALIES: usize = 128;
const DEFAULT_MAX_REPORTED_STRUCTURE_OCCURRENCES: usize = 512;
const DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM: usize = 32;
const DEFAULT_MAX_POSITIONS_PER_PATTERN: usize = 24;
const DEFAULT_CONTEXT_RADIUS: usize = 48;
const MAX_SAFE_TENSOR_HEADER: usize = 16 * 1024 * 1024;
const PARALLEL_THRESHOLD_BYTES: u64 = 4 * 1024 * 1024;
const HIGH_ENTROPY_THRESHOLD: f64 = 7.85;
const LOW_ENTROPY_THRESHOLD: f64 = 0.20;
const ENTROPY_TRANSITION_DELTA: f64 = 2.25;
const ZERO_RUN_ANOMALY_THRESHOLD: u64 = 4096;
const STREAMING_KEYWORD_TAIL: usize = 31;

const STRONG_MODEL_CONTEXT_KEYS: &[&[u8]] = &[
    b"general.architecture",
    b"general.name",
    b"model_type",
    b"architectures",
    b"architecture",
    b"_name_or_path",
    b"model_name",
    b"model_id",
    b"model-id",
];

const MODEL_SHAPE_KEYS: &[&[u8]] = &[
    b"hidden_size",
    b"num_hidden_layers",
    b"num_attention_heads",
    b"vocab_size",
    b"intermediate_size",
    b"embedding_length",
    b"context_length",
    b"block_count",
];

const QUANT_CONTEXT_KEYS: &[&[u8]] = &[
    b"quant",
    b"quantized",
    b"dtype",
    b"type",
    b"tensor",
    b"weight",
    b"weights",
    b"format",
    b"bits",
    b"gguf",
    b"ggml",
];

const GENERIC_STRUCTURED_KEYS: &[&[u8]] = &[
    b"model_type",
    b"architectures",
    b"hidden_size",
    b"num_hidden_layers",
    b"num_attention_heads",
    b"vocab_size",
    b"tokenizer_class",
    b"dataset_info",
    b"splits",
    b"dtype",
    b"data_offsets",
];

const DATASET_CONTEXT_KEYS: &[&[u8]] = &[
    b"dataset",
    b"splits",
    b"download",
    b"rows",
    b"samples",
    b"examples",
    b"train",
    b"validation",
    b"test",
    b"tokens",
];

#[derive(Debug)]
pub enum DetectError {
    Io(io::Error),
    InvalidPath,
    InvalidUtf8Path,
}

impl fmt::Display for DetectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetectError::Io(e) => write!(f, "io error: {}", e),
            DetectError::InvalidPath => write!(f, "invalid path"),
            DetectError::InvalidUtf8Path => write!(f, "path is not valid UTF-8"),
        }
    }
}

impl std::error::Error for DetectError {}

impl From<io::Error> for DetectError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Debug, Clone)]
pub struct AnalysisOptions {
    pub parallel: bool,
    pub pretty_json: bool,
    pub scan_window_bytes: usize,
    pub max_safetensors_header_bytes: usize,
    pub deep_scan_chunk_bytes: usize,
    pub deep_scan_overlap_bytes: usize,
    pub deep_entropy_window_bytes: usize,
    pub max_reported_shapes: usize,
    pub max_reported_structure_occurrences: usize,
    pub max_pattern_matches_per_item: usize,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            parallel: true,
            pretty_json: false,
            scan_window_bytes: DEFAULT_SCAN_WINDOW,
            max_safetensors_header_bytes: MAX_SAFE_TENSOR_HEADER,
            deep_scan_chunk_bytes: DEFAULT_DEEP_SCAN_CHUNK_BYTES,
            deep_scan_overlap_bytes: DEFAULT_DEEP_SCAN_OVERLAP_BYTES,
            deep_entropy_window_bytes: DEFAULT_DEEP_ENTROPY_WINDOW_BYTES,
            max_reported_shapes: DEFAULT_MAX_REPORTED_ANOMALIES,
            max_reported_structure_occurrences: DEFAULT_MAX_REPORTED_STRUCTURE_OCCURRENCES,
            max_pattern_matches_per_item: DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern: String,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone)]
pub struct Analysis {
    pub ok: bool,
    pub file_name: String,
    pub created_at_utc: String,
    pub byte_count: u64,
    pub scanned_byte_count: u64,
    pub is_probably_text: bool,
    pub entropy_sample: f64,
    pub signatures: Vec<String>,
    pub detected_specs: Vec<SpecDetection>,
    pub detected_models: Vec<ModelHint>,
    pub matched_patterns: Vec<MatchedPatternGroup>,
    pub detected_data_structures: Vec<DataStructureHint>,
    pub quantization: Vec<QuantizationHint>,
    pub dataset_size: Vec<DatasetSizeHint>,
    pub parameter_data: Vec<ParameterHint>,
    pub shapes: Vec<EntroshapeHint>,
    pub metadata: BTreeMap<String, String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SpecDetection {
    pub name: String,
    pub version: Option<String>,
    pub source: String,
    pub notes: Vec<String>,
    pub pattern_matches: Vec<PatternMatch>,
}

#[derive(Debug, Clone)]
pub struct ModelHint {
    pub family: String,
    pub variant: Option<String>,
    pub source: String,
    pub pattern_matches: Vec<PatternMatch>,
}

#[derive(Debug, Clone)]
pub struct MatchedPatternGroup {
    pub category: String,
    pub source: String,
    pub pattern_matches: Vec<PatternMatch>,
}

#[derive(Debug, Clone)]
pub struct DataStructureHint {
    pub name: String,
    pub source: String,
    pub offset: Option<u64>,
    pub length: Option<u64>,
    pub details: BTreeMap<String, String>,
    pub pattern_matches: Vec<PatternMatch>,
}

#[derive(Debug, Clone)]
pub struct QuantizationHint {
    pub scheme: String,
    pub source: String,
    pub pattern_matches: Vec<PatternMatch>,
}

#[derive(Debug, Clone)]
pub struct DatasetSizeHint {
    pub metric: String,
    pub value: String,
    pub source: String,
    pub pattern_matches: Vec<PatternMatch>,
}

#[derive(Debug, Clone)]
pub struct ParameterHint {
    pub metric: String,
    pub value: String,
    pub source: String,
    pub pattern_matches: Vec<PatternMatch>,
}

#[derive(Debug, Clone)]
pub struct EntroshapeHint {
    pub kind: String,
    pub offset: u64,
    pub length: Option<u64>,
    pub description: String,
    pub source: String,
    pub details: BTreeMap<String, String>,
    pub pattern_matches: Vec<PatternMatch>,
}

#[derive(Debug, Default)]
struct DetectorOutput {
    specs: Vec<SpecDetection>,
    models: Vec<ModelHint>,
    matched_patterns: Vec<MatchedPatternGroup>,
    data_structures: Vec<DataStructureHint>,
    quantization: Vec<QuantizationHint>,
    dataset_size: Vec<DatasetSizeHint>,
    parameter_data: Vec<ParameterHint>,
    shapes: Vec<EntroshapeHint>,
    metadata: BTreeMap<String, String>,
    warnings: Vec<String>,
}

impl DetectorOutput {
    fn merge_from(&mut self, other: DetectorOutput) {
        self.specs.extend(other.specs);
        self.models.extend(other.models);
        self.matched_patterns.extend(other.matched_patterns);
        self.data_structures.extend(other.data_structures);
        self.quantization.extend(other.quantization);
        self.dataset_size.extend(other.dataset_size);
        self.parameter_data.extend(other.parameter_data);
        self.shapes.extend(other.shapes);
        self.metadata.extend(other.metadata);
        self.warnings.extend(other.warnings);
    }

    fn merge_into(self, analysis: &mut Analysis, options: &AnalysisOptions) {
        for spec in self.specs {
            push_or_merge_spec(
                &mut analysis.detected_specs,
                spec,
                options.max_pattern_matches_per_item,
            );
        }
        for model in self.models {
            push_or_merge_model(
                &mut analysis.detected_models,
                model,
                options.max_pattern_matches_per_item,
            );
        }
        for group in self.matched_patterns {
            push_or_merge_matched_pattern_group(
                &mut analysis.matched_patterns,
                group,
                options.max_pattern_matches_per_item,
            );
        }
        for ds in self.data_structures {
            push_limited_data_structure(analysis, ds, options);
        }
        for quant in self.quantization {
            push_or_merge_quantization(
                &mut analysis.quantization,
                quant,
                options.max_pattern_matches_per_item,
            );
        }
        for dataset in self.dataset_size {
            push_or_merge_dataset(
                &mut analysis.dataset_size,
                dataset,
                options.max_pattern_matches_per_item,
            );
        }
        for param in self.parameter_data {
            push_or_merge_parameter(
                &mut analysis.parameter_data,
                param,
                options.max_pattern_matches_per_item,
            );
        }
        for shaper in self.shapes {
            push_limited_shaper(analysis, shaper, options);
        }
        for (k, v) in self.metadata {
            analysis.metadata.entry(k).or_insert(v);
        }
        for warning in self.warnings {
            push_unique_string(&mut analysis.warnings, warning);
        }
    }
}

#[derive(Copy, Clone)]
enum ScanMode {
    Prefix,
    Deep,
}

#[derive(Copy, Clone)]
enum MatchDomain {
    FormatSpecific,
    Structured,
    Generic,
}

#[derive(Copy, Clone)]
enum TokenBoundaryMode {
    AlphaNum,
    AlphaNumUnderscore,
}

struct SharedScanContext {
    lower_scan: Vec<u8>,
    file_name_lower: Vec<u8>,
    jsonish_prefix: bool,
    is_probably_text: bool,
    structured_marker_count: usize,
}

impl SharedScanContext {
    fn new(file_name: &str, scan_bytes: &[u8]) -> Self {
        let lower_scan = ascii_lower_vec(scan_bytes);
        let file_name_lower = ascii_lower_vec(file_name.as_bytes());
        let jsonish_prefix = is_jsonish_text_prefix(scan_bytes);
        let is_probably_text = is_probably_text(scan_bytes);
        let structured_marker_count = count_present_patterns(&lower_scan, GENERIC_STRUCTURED_KEYS);
        Self {
            lower_scan,
            file_name_lower,
            jsonish_prefix,
            is_probably_text,
            structured_marker_count,
        }
    }
}

struct LoadedPrefix {
    file_name: String,
    total_byte_count: u64,
    scan_bytes: Vec<u8>,
}

struct DeepScanState {
    tail: Vec<u8>,
    previous_entropy: Option<f64>,
    zero_run_start: Option<u64>,
    zero_run_length: u64,
    root_safetensors: Option<StreamingSafetensorsHeaderState>,
}

impl DeepScanState {
    fn new() -> Self {
        Self {
            tail: Vec::new(),
            previous_entropy: None,
            zero_run_start: None,
            zero_run_length: 0,
            root_safetensors: None,
        }
    }
}

struct StreamingSafetensorsHeaderState {
    declared_header_bytes: u64,
    processed_header_bytes: u64,
    saw_open_brace: bool,
    saw_close_brace: bool,
    keyword_tail: Vec<u8>,
    metadata_pos: Option<u64>,
    dtype_pos: Option<u64>,
    data_offsets_pos: Option<u64>,
}

impl StreamingSafetensorsHeaderState {
    fn new(declared_header_bytes: u64) -> Self {
        Self {
            declared_header_bytes,
            processed_header_bytes: 0,
            saw_open_brace: false,
            saw_close_brace: false,
            keyword_tail: Vec::new(),
            metadata_pos: None,
            dtype_pos: None,
            data_offsets_pos: None,
        }
    }
}

pub fn analyze_file_json<P: AsRef<Path>>(path: P) -> String {
    analyze_file_json_with_options(path, &AnalysisOptions::default())
}

pub fn analyze_file_json_pretty<P: AsRef<Path>>(path: P) -> String {
    let mut options = AnalysisOptions::default();
    options.pretty_json = true;
    analyze_file_json_with_options(path, &options)
}

pub fn analyze_file_json_with_options<P: AsRef<Path>>(
    path: P,
    options: &AnalysisOptions,
) -> String {
    let path_string = path.as_ref().to_string_lossy().into_owned();
    match panic::catch_unwind(AssertUnwindSafe(|| {
        analyze_file_internal(path.as_ref(), options, ScanMode::Prefix)
    })) {
        Ok(Ok(analysis)) => analysis_to_json(&analysis, options.pretty_json),
        Ok(Err(err)) => error_to_json(
            "analysis_error",
            &err.to_string(),
            &path_string,
            options.pretty_json,
        ),
        Err(payload) => error_to_json(
            "panic",
            &panic_message(payload.as_ref()),
            &path_string,
            options.pretty_json,
        ),
    }
}

pub fn analyze_file_json_deep<P: AsRef<Path>>(path: P) -> String {
    analyze_file_json_deep_with_options(path, &AnalysisOptions::default())
}

pub fn analyze_file_json_deep_pretty<P: AsRef<Path>>(path: P) -> String {
    let mut options = AnalysisOptions::default();
    options.pretty_json = true;
    analyze_file_json_deep_with_options(path, &options)
}

pub fn analyze_file_json_deep_with_options<P: AsRef<Path>>(
    path: P,
    options: &AnalysisOptions,
) -> String {
    let path_string = path.as_ref().to_string_lossy().into_owned();
    match panic::catch_unwind(AssertUnwindSafe(|| {
        analyze_file_internal(path.as_ref(), options, ScanMode::Deep)
    })) {
        Ok(Ok(analysis)) => analysis_to_json(&analysis, options.pretty_json),
        Ok(Err(err)) => error_to_json(
            "analysis_error",
            &err.to_string(),
            &path_string,
            options.pretty_json,
        ),
        Err(payload) => error_to_json(
            "panic",
            &panic_message(payload.as_ref()),
            &path_string,
            options.pretty_json,
        ),
    }
}

pub fn analyze_bytes_json(file_name: &str, bytes: &[u8]) -> String {
    analyze_bytes_json_with_options(file_name, bytes, &AnalysisOptions::default())
}

pub fn analyze_bytes_json_pretty(file_name: &str, bytes: &[u8]) -> String {
    let mut options = AnalysisOptions::default();
    options.pretty_json = true;
    analyze_bytes_json_with_options(file_name, bytes, &options)
}

pub fn analyze_bytes_json_with_options(
    file_name: &str,
    bytes: &[u8],
    options: &AnalysisOptions,
) -> String {
    match panic::catch_unwind(AssertUnwindSafe(|| {
        analyze_bytes_internal(file_name, bytes, options, ScanMode::Prefix)
    })) {
        Ok(analysis) => analysis_to_json(&analysis, options.pretty_json),
        Err(payload) => error_to_json(
            "panic",
            &panic_message(payload.as_ref()),
            file_name,
            options.pretty_json,
        ),
    }
}

pub fn analyze_bytes_json_deep(file_name: &str, bytes: &[u8]) -> String {
    analyze_bytes_json_deep_with_options(file_name, bytes, &AnalysisOptions::default())
}

pub fn analyze_bytes_json_deep_pretty(file_name: &str, bytes: &[u8]) -> String {
    let mut options = AnalysisOptions::default();
    options.pretty_json = true;
    analyze_bytes_json_deep_with_options(file_name, bytes, &options)
}

pub fn analyze_bytes_json_deep_with_options(
    file_name: &str,
    bytes: &[u8],
    options: &AnalysisOptions,
) -> String {
    match panic::catch_unwind(AssertUnwindSafe(|| {
        analyze_bytes_internal(file_name, bytes, options, ScanMode::Deep)
    })) {
        Ok(analysis) => analysis_to_json(&analysis, options.pretty_json),
        Err(payload) => error_to_json(
            "panic",
            &panic_message(payload.as_ref()),
            file_name,
            options.pretty_json,
        ),
    }
}

fn analyze_file_internal(
    path: &Path,
    options: &AnalysisOptions,
    mode: ScanMode,
) -> Result<Analysis, DetectError> {
    match mode {
        ScanMode::Prefix => {
            let loaded = load_file_prefix(path, options)?;
            Ok(analyze_prefix_bytes_internal(
                &loaded.file_name,
                &loaded.scan_bytes,
                loaded.total_byte_count,
                options,
            ))
        }
        ScanMode::Deep => analyze_file_deep_internal(path, options),
    }
}

fn analyze_bytes_internal(
    file_name: &str,
    bytes: &[u8],
    options: &AnalysisOptions,
    mode: ScanMode,
) -> Analysis {
    match mode {
        ScanMode::Prefix => {
            analyze_prefix_bytes_internal(file_name, bytes, bytes.len() as u64, options)
        }
        ScanMode::Deep => analyze_deep_buffer_internal(file_name, bytes, options),
    }
}

fn load_file_prefix(path: &Path, options: &AnalysisOptions) -> Result<LoadedPrefix, DetectError> {
    let file_name = path
        .file_name()
        .ok_or(DetectError::InvalidPath)?
        .to_str()
        .ok_or(DetectError::InvalidUtf8Path)?
        .to_string();

    let mut file = File::open(path)?;
    let total_byte_count = file.metadata()?.len();
    let mut initial_target = options.scan_window_bytes.max(9) as u64;
    if total_byte_count < initial_target {
        initial_target = total_byte_count;
    }

    let mut scan_bytes = Vec::with_capacity(initial_target as usize);
    file.by_ref()
        .take(initial_target)
        .read_to_end(&mut scan_bytes)?;

    if scan_bytes.len() >= 9 {
        if let Some(header_len) = read_le_u64(&scan_bytes[..8]) {
            let looks_like_safetensors = header_len > 0 && scan_bytes[8] == b'{';
            if looks_like_safetensors && header_len <= options.max_safetensors_header_bytes as u64 {
                let required = 8u64.saturating_add(header_len).min(total_byte_count);
                if required > scan_bytes.len() as u64 {
                    let additional = required - scan_bytes.len() as u64;
                    scan_bytes.reserve(additional as usize);
                    file.take(additional).read_to_end(&mut scan_bytes)?;
                }
            }
        }
    }

    Ok(LoadedPrefix {
        file_name,
        total_byte_count,
        scan_bytes,
    })
}

fn analyze_prefix_bytes_internal(
    file_name: &str,
    scan_bytes: &[u8],
    total_byte_count: u64,
    options: &AnalysisOptions,
) -> Analysis {
    let ctx = SharedScanContext::new(file_name, scan_bytes);
    let mut analysis = Analysis {
        ok: true,
        file_name: file_name.to_string(),
        created_at_utc: now_utc_iso(),
        byte_count: total_byte_count,
        scanned_byte_count: scan_bytes.len() as u64,
        is_probably_text: ctx.is_probably_text,
        entropy_sample: estimate_entropy_sample(scan_bytes, ENTROPY_SAMPLE_MAX),
        signatures: detect_signatures(scan_bytes),
        detected_specs: Vec::new(),
        detected_models: Vec::new(),
        matched_patterns: Vec::new(),
        detected_data_structures: Vec::new(),
        quantization: Vec::new(),
        dataset_size: Vec::new(),
        parameter_data: Vec::new(),
        shapes: Vec::new(),
        metadata: BTreeMap::new(),
        warnings: Vec::new(),
    };

    analysis.metadata.insert(
        "scan_strategy".into(),
        if analysis.scanned_byte_count < analysis.byte_count {
            "prefix_window".into()
        } else {
            "full_buffer".into()
        },
    );
    analysis.metadata.insert(
        "configured_scan_window_bytes".into(),
        options.scan_window_bytes.to_string(),
    );
    analysis.metadata.insert(
        "configured_max_safetensors_header_bytes".into(),
        options.max_safetensors_header_bytes.to_string(),
    );
    analysis
        .metadata
        .insert("pretty_json".into(), options.pretty_json.to_string());
    if analysis.scanned_byte_count < analysis.byte_count {
        analysis
            .metadata
            .insert("partial_scan".into(), "true".into());
    }

    run_window_detectors(
        scan_bytes,
        &ctx,
        options,
        &mut analysis,
        0,
        true,
        ScanMode::Prefix,
    );
    finalize_analysis(&mut analysis);
    analysis
}

fn analyze_file_deep_internal(
    path: &Path,
    options: &AnalysisOptions,
) -> Result<Analysis, DetectError> {
    let file_name = path
        .file_name()
        .ok_or(DetectError::InvalidPath)?
        .to_str()
        .ok_or(DetectError::InvalidUtf8Path)?
        .to_string();

    let mut file = File::open(path)?;
    let total_byte_count = file.metadata()?.len();
    let chunk_size = options.deep_scan_chunk_bytes.max(4096);
    let overlap_size = options.deep_scan_overlap_bytes.min(chunk_size / 2).max(64);

    let mut first_chunk = vec![0u8; chunk_size];
    let first_read = file.read(&mut first_chunk)?;
    first_chunk.truncate(first_read);

    let mut analysis = Analysis {
        ok: true,
        file_name: file_name.clone(),
        created_at_utc: now_utc_iso(),
        byte_count: total_byte_count,
        scanned_byte_count: total_byte_count,
        is_probably_text: is_probably_text(&first_chunk),
        entropy_sample: estimate_entropy_sample(&first_chunk, ENTROPY_SAMPLE_MAX),
        signatures: detect_signatures(&first_chunk),
        detected_specs: Vec::new(),
        detected_models: Vec::new(),
        matched_patterns: Vec::new(),
        detected_data_structures: Vec::new(),
        quantization: Vec::new(),
        dataset_size: Vec::new(),
        parameter_data: Vec::new(),
        shapes: Vec::new(),
        metadata: BTreeMap::new(),
        warnings: Vec::new(),
    };

    analysis
        .metadata
        .insert("scan_strategy".into(), "deep_stream".into());
    analysis.metadata.insert(
        "configured_deep_scan_chunk_bytes".into(),
        chunk_size.to_string(),
    );
    analysis.metadata.insert(
        "configured_deep_scan_overlap_bytes".into(),
        overlap_size.to_string(),
    );
    analysis.metadata.insert(
        "configured_deep_entropy_window_bytes".into(),
        options.deep_entropy_window_bytes.to_string(),
    );
    analysis
        .metadata
        .insert("pretty_json".into(), options.pretty_json.to_string());

    let mut state = DeepScanState::new();
    let mut offset = 0u64;

    if !first_chunk.is_empty() {
        process_deep_chunk(
            &file_name,
            &first_chunk,
            offset,
            &mut state,
            &mut analysis,
            options,
            overlap_size,
        );
        offset += first_chunk.len() as u64;
    }

    let mut buffer = vec![0u8; chunk_size];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        process_deep_chunk(
            &file_name,
            &buffer[..read],
            offset,
            &mut state,
            &mut analysis,
            options,
            overlap_size,
        );
        offset += read as u64;
    }

    finalize_deep_state(&mut state, &mut analysis, options);
    finalize_analysis(&mut analysis);
    Ok(analysis)
}

fn analyze_deep_buffer_internal(
    file_name: &str,
    bytes: &[u8],
    options: &AnalysisOptions,
) -> Analysis {
    let chunk_size = options.deep_scan_chunk_bytes.max(4096);
    let overlap_size = options.deep_scan_overlap_bytes.min(chunk_size / 2).max(64);
    let sample = &bytes[..bytes.len().min(chunk_size)];
    let mut analysis = Analysis {
        ok: true,
        file_name: file_name.to_string(),
        created_at_utc: now_utc_iso(),
        byte_count: bytes.len() as u64,
        scanned_byte_count: bytes.len() as u64,
        is_probably_text: is_probably_text(sample),
        entropy_sample: estimate_entropy_sample(sample, ENTROPY_SAMPLE_MAX),
        signatures: detect_signatures(sample),
        detected_specs: Vec::new(),
        detected_models: Vec::new(),
        matched_patterns: Vec::new(),
        detected_data_structures: Vec::new(),
        quantization: Vec::new(),
        dataset_size: Vec::new(),
        parameter_data: Vec::new(),
        shapes: Vec::new(),
        metadata: BTreeMap::new(),
        warnings: Vec::new(),
    };

    analysis
        .metadata
        .insert("scan_strategy".into(), "deep_buffer".into());
    analysis.metadata.insert(
        "configured_deep_scan_chunk_bytes".into(),
        chunk_size.to_string(),
    );
    analysis.metadata.insert(
        "configured_deep_scan_overlap_bytes".into(),
        overlap_size.to_string(),
    );
    analysis.metadata.insert(
        "configured_deep_entropy_window_bytes".into(),
        options.deep_entropy_window_bytes.to_string(),
    );
    analysis
        .metadata
        .insert("pretty_json".into(), options.pretty_json.to_string());

    let mut state = DeepScanState::new();
    let mut offset = 0usize;
    while offset < bytes.len() {
        let end = bytes.len().min(offset + chunk_size);
        process_deep_chunk(
            file_name,
            &bytes[offset..end],
            offset as u64,
            &mut state,
            &mut analysis,
            options,
            overlap_size,
        );
        offset = end;
    }

    finalize_deep_state(&mut state, &mut analysis, options);
    finalize_analysis(&mut analysis);
    analysis
}

fn process_deep_chunk(
    file_name: &str,
    chunk: &[u8],
    chunk_offset: u64,
    state: &mut DeepScanState,
    analysis: &mut Analysis,
    options: &AnalysisOptions,
    overlap_size: usize,
) {
    if chunk.is_empty() {
        return;
    }

    maybe_start_streaming_safetensors_header(state, chunk, chunk_offset, options, analysis);

    let tail_len = state.tail.len();
    let window: Cow<'_, [u8]> = if tail_len == 0 {
        Cow::Borrowed(chunk)
    } else {
        let mut combined = Vec::with_capacity(tail_len + chunk.len());
        combined.extend_from_slice(&state.tail);
        combined.extend_from_slice(chunk);
        Cow::Owned(combined)
    };
    let window_base_offset = chunk_offset.saturating_sub(tail_len as u64);

    scan_embedded_headers(window.as_ref(), window_base_offset, analysis, options);

    if chunk_offset == 0 || is_likely_textual_or_mixed(window.as_ref()) {
        let ctx = SharedScanContext::new(file_name, window.as_ref());
        run_window_detectors(
            window.as_ref(),
            &ctx,
            options,
            analysis,
            window_base_offset,
            chunk_offset == 0,
            ScanMode::Deep,
        );
    }

    scan_entropy_windows(chunk, chunk_offset, state, analysis, options);
    scan_zero_runs(chunk, chunk_offset, state, analysis, options);
    stream_safetensors_header_keywords(state, chunk, chunk_offset, analysis, options);

    let keep = overlap_size.min(window.len());
    state.tail.clear();
    state.tail.extend_from_slice(&window[window.len() - keep..]);
}

fn finalize_deep_state(
    state: &mut DeepScanState,
    analysis: &mut Analysis,
    options: &AnalysisOptions,
) {
    if state.zero_run_length >= ZERO_RUN_ANOMALY_THRESHOLD {
        let mut details = BTreeMap::new();
        details.insert("zero_run_bytes".into(), state.zero_run_length.to_string());
        let offset = state.zero_run_start.unwrap_or(0);
        push_limited_shaper(
            analysis,
            EntroshapeHint {
                kind: "long_zero_run".into(),
                offset,
                length: Some(state.zero_run_length),
                description: format!("zero-byte run of {} bytes", state.zero_run_length),
                source: "deep zero-run scan".into(),
                details,
                pattern_matches: Vec::new(),
            },
            options,
        );
    }

    if state.root_safetensors.is_some() {
        push_unique_string(
            &mut analysis.warnings,
            "deep safetensors streaming parser reached EOF before the declared header completed"
                .into(),
        );
    }
}

fn should_defer_root_safetensors_to_deep_stream(
    scan_bytes: &[u8],
    options: &AnalysisOptions,
) -> bool {
    if scan_bytes.len() < 9 || scan_bytes[8] != b'{' {
        return false;
    }

    let Some(header_len) = read_le_u64(&scan_bytes[..8]) else {
        return false;
    };
    if header_len == 0 {
        return false;
    }

    header_len > options.max_safetensors_header_bytes as u64
        || 8u64.saturating_add(header_len) > scan_bytes.len() as u64
}

fn run_window_detectors(
    scan_bytes: &[u8],
    ctx: &SharedScanContext,
    options: &AnalysisOptions,
    analysis: &mut Analysis,
    base_offset: u64,
    allow_root_prefix: bool,
    scan_mode: ScanMode,
) {
    let defer_root_safetensors = allow_root_prefix
        && matches!(scan_mode, ScanMode::Deep)
        && base_offset == 0
        && should_defer_root_safetensors_to_deep_stream(scan_bytes, options);

    if options.parallel && should_parallelize(scan_bytes.len() as u64) {
        std::thread::scope(|scope| {
            let root_handle = scope.spawn(|| {
                let mut batch = DetectorOutput::default();
                if allow_root_prefix {
                    batch.merge_from(detect_gguf(scan_bytes, ctx, base_offset));
                    if !defer_root_safetensors {
                        batch.merge_from(detect_safetensors(
                            scan_bytes,
                            ctx,
                            base_offset,
                            options,
                            scan_mode,
                        ));
                    }
                    batch.merge_from(detect_zip_based(scan_bytes, ctx, base_offset));
                    batch.merge_from(detect_hdf5(scan_bytes, ctx, base_offset));
                }
                batch
            });

            let model_handle = scope.spawn(|| {
                let mut batch = DetectorOutput::default();
                batch.merge_from(detect_onnx(ctx, base_offset));
                batch.merge_from(detect_json_structures(ctx, base_offset));
                batch.merge_from(detect_generic_hints(ctx, base_offset));
                batch
            });

            let r1 = root_handle.join();
            let r2 = model_handle.join();
            for result in [r1, r2] {
                match result {
                    Ok(batch) => batch.merge_into(analysis, options),
                    Err(payload) => push_unique_string(
                        &mut analysis.warnings,
                        format!(
                            "detector worker panicked: {}",
                            panic_message(payload.as_ref())
                        ),
                    ),
                }
            }
        });
    } else {
        let mut batch = DetectorOutput::default();
        if allow_root_prefix {
            batch.merge_from(detect_gguf(scan_bytes, ctx, base_offset));
            if !defer_root_safetensors {
                batch.merge_from(detect_safetensors(
                    scan_bytes,
                    ctx,
                    base_offset,
                    options,
                    scan_mode,
                ));
            }
            batch.merge_from(detect_zip_based(scan_bytes, ctx, base_offset));
            batch.merge_from(detect_hdf5(scan_bytes, ctx, base_offset));
        }
        batch.merge_from(detect_onnx(ctx, base_offset));
        batch.merge_from(detect_json_structures(ctx, base_offset));
        batch.merge_from(detect_generic_hints(ctx, base_offset));
        batch.merge_into(analysis, options);
    }
}

fn detect_signatures(scan_bytes: &[u8]) -> Vec<String> {
    let mut out = Vec::with_capacity(6);
    if has_prefix(scan_bytes, b"GGUF") {
        out.push("magic:GGUF".into());
    }
    if has_prefix(scan_bytes, b"PK\x03\x04") {
        out.push("magic:ZIP".into());
    }
    if has_prefix(scan_bytes, b"\x89HDF\r\n\x1a\n") {
        out.push("magic:HDF5".into());
    }
    if matches!(
        first_non_whitespace_byte(scan_bytes),
        Some(b'{') | Some(b'[')
    ) {
        out.push("magic:JSON-ish".into());
    }
    if memmem(scan_bytes, b"safetensors").is_some() {
        out.push("text:safetensors".into());
    }
    if memmem(scan_bytes, b"onnx").is_some() {
        out.push("text:onnx".into());
    }
    out.sort();
    out.dedup();
    out
}

fn detect_gguf(scan_bytes: &[u8], ctx: &SharedScanContext, base_offset: u64) -> DetectorOutput {
    let mut out = DetectorOutput::default();
    if !has_prefix(scan_bytes, b"GGUF") {
        return out;
    }

    let mut notes = Vec::new();
    let version = read_le_u32(scan_bytes.get(4..8).unwrap_or_default()).map(|v| v.to_string());
    let magic_match = PatternMatch {
        pattern: "GGUF".into(),
        offset: base_offset,
        length: 4,
    };

    let mut details = BTreeMap::new();
    if let Some(tensor_count) = read_le_u64(scan_bytes.get(8..16).unwrap_or_default()) {
        out.metadata
            .insert("gguf_tensor_count".into(), tensor_count.to_string());
        details.insert("tensor_count".into(), tensor_count.to_string());
        notes.push(format!("tensor_count={}", tensor_count));
    }
    if let Some(kv_count) = read_le_u64(scan_bytes.get(16..24).unwrap_or_default()) {
        out.metadata
            .insert("gguf_kv_count".into(), kv_count.to_string());
        details.insert("kv_count".into(), kv_count.to_string());
        notes.push(format!("kv_count={}", kv_count));
    }
    if scan_bytes.len() < 24 {
        out.warnings.push("GGUF header appears truncated".into());
    }

    out.specs.push(SpecDetection {
        name: "GGUF".into(),
        version,
        source: "binary header".into(),
        notes,
        pattern_matches: vec![magic_match.clone()],
    });

    out.data_structures.push(DataStructureHint {
        name: "gguf_header".into(),
        source: "binary header".into(),
        offset: Some(base_offset),
        length: Some(24),
        details,
        pattern_matches: vec![magic_match],
    });

    let arch_matches = collect_key_matches(
        &ctx.lower_scan,
        &[
            b"general.architecture",
            b"llama.context_length",
            b"llama.embedding_length",
        ],
        base_offset,
        DEFAULT_MAX_POSITIONS_PER_PATTERN,
    );
    if !arch_matches.is_empty() {
        out.data_structures.push(DataStructureHint {
            name: "gguf_kv_region".into(),
            source: "GGUF key/value strings".into(),
            offset: Some(arch_matches[0].offset),
            length: None,
            details: BTreeMap::new(),
            pattern_matches: arch_matches,
        });
    }

    extract_common_model_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::FormatSpecific,
        "GGUF text region",
        &mut out,
    );
    extract_quantization_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::FormatSpecific,
        "GGUF text region",
        &mut out,
    );
    extract_parameter_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::FormatSpecific,
        "GGUF text region",
        &mut out,
    );
    out
}

fn detect_safetensors(
    scan_bytes: &[u8],
    _ctx: &SharedScanContext,
    base_offset: u64,
    options: &AnalysisOptions,
    scan_mode: ScanMode,
) -> DetectorOutput {
    let mut out = DetectorOutput::default();
    if scan_bytes.len() < 9 {
        return out;
    }

    let Some(header_len) = read_le_u64(&scan_bytes[..8]) else {
        return out;
    };
    if header_len == 0 {
        return out;
    }
    if header_len > options.max_safetensors_header_bytes as u64 {
        if matches!(scan_mode, ScanMode::Prefix) {
            out.warnings.push(format!(
                "safetensors header length {} exceeds configured max {}; skipping detailed parse",
                header_len, options.max_safetensors_header_bytes
            ));
        }
        return out;
    }

    let end_u64 = 8u64.saturating_add(header_len);
    if end_u64 > scan_bytes.len() as u64 {
        if matches!(scan_mode, ScanMode::Prefix) {
            out.warnings
                .push("safetensors header appears truncated in scanned window".into());
        }
        return out;
    }

    let end = end_u64 as usize;
    let header = &scan_bytes[8..end];
    if !(header.starts_with(b"{") && header.ends_with(b"}")) {
        return out;
    }

    let header_lower = ascii_lower_vec(header);
    let metadata_matches =
        collect_key_matches(&header_lower, &[b"\"__metadata__\""], base_offset + 8, 4);
    let dtype_matches = collect_key_matches(&header_lower, &[b"\"dtype\""], base_offset + 8, 8);
    let offsets_matches =
        collect_key_matches(&header_lower, &[b"\"data_offsets\""], base_offset + 8, 8);

    let mut score = 2;
    if !metadata_matches.is_empty() {
        score += 3;
    }
    if !dtype_matches.is_empty() {
        score += 2;
    }
    if !offsets_matches.is_empty() {
        score += 2;
    }

    if score < 5 {
        return out;
    }

    let mut pattern_matches = Vec::new();
    extend_pattern_matches_limited(
        &mut pattern_matches,
        metadata_matches.clone(),
        options.max_pattern_matches_per_item,
    );
    extend_pattern_matches_limited(
        &mut pattern_matches,
        dtype_matches.clone(),
        options.max_pattern_matches_per_item,
    );
    extend_pattern_matches_limited(
        &mut pattern_matches,
        offsets_matches.clone(),
        options.max_pattern_matches_per_item,
    );

    out.specs.push(SpecDetection {
        name: "safetensors".into(),
        version: None,
        source: "streamed header JSON".into(),
        notes: vec![format!("header_bytes={}", header_len)],
        pattern_matches: pattern_matches.clone(),
    });

    let mut details = BTreeMap::new();
    details.insert("header_bytes".into(), header_len.to_string());
    out.data_structures.push(DataStructureHint {
        name: "tensor_index".into(),
        source: "safetensors header".into(),
        offset: Some(base_offset + 8),
        length: Some(header_len),
        details,
        pattern_matches,
    });

    out.metadata
        .insert("safetensors_header_bytes".into(), header_len.to_string());
    extract_common_model_hints(
        &header_lower,
        base_offset + 8,
        MatchDomain::Structured,
        "safetensors header",
        &mut out,
    );
    extract_quantization_hints(
        &header_lower,
        base_offset + 8,
        MatchDomain::Structured,
        "safetensors header",
        &mut out,
    );
    extract_parameter_hints(
        &header_lower,
        base_offset + 8,
        MatchDomain::Structured,
        "safetensors header",
        &mut out,
    );
    out
}

fn detect_zip_based(
    scan_bytes: &[u8],
    ctx: &SharedScanContext,
    base_offset: u64,
) -> DetectorOutput {
    let mut out = DetectorOutput::default();
    if !has_prefix(scan_bytes, b"PK\x03\x04") {
        return out;
    }

    let zip_magic = PatternMatch {
        pattern: "PK\\x03\\x04".into(),
        offset: base_offset,
        length: 4,
    };
    let pytorch_hits = collect_key_matches(
        &ctx.lower_scan,
        &[b"data.pkl", b"archive/", b"pytorch", b"model_weights"],
        base_offset,
        DEFAULT_MAX_POSITIONS_PER_PATTERN,
    );
    let numpy_hits = collect_key_matches(
        &ctx.lower_scan,
        &[b".npy", b"numpy", b"npz"],
        base_offset,
        DEFAULT_MAX_POSITIONS_PER_PATTERN,
    );

    let pytorch_score = pytorch_hits.len() as i32
        + if ctx.file_name_lower.ends_with(b".pt") || ctx.file_name_lower.ends_with(b".pth") {
            2
        } else {
            0
        };
    let npz_score = numpy_hits.len() as i32
        + if ctx.file_name_lower.ends_with(b".npz") {
            2
        } else {
            0
        };

    if pytorch_score >= 2 {
        let mut matches = vec![zip_magic.clone()];
        extend_pattern_matches_limited(
            &mut matches,
            pytorch_hits.clone(),
            DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
        );
        out.specs.push(SpecDetection {
            name: "PyTorch ZIP artifact".into(),
            version: None,
            source: "ZIP member-name hints".into(),
            notes: vec!["ZIP container with PyTorch-like entries".into()],
            pattern_matches: matches.clone(),
        });
        out.data_structures.push(DataStructureHint {
            name: "pickle_payload".into(),
            source: "ZIP member-name hints".into(),
            offset: Some(base_offset),
            length: None,
            details: BTreeMap::new(),
            pattern_matches: matches,
        });
    }

    if npz_score >= 2 {
        let mut matches = vec![zip_magic.clone()];
        extend_pattern_matches_limited(
            &mut matches,
            numpy_hits.clone(),
            DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
        );
        out.specs.push(SpecDetection {
            name: "NPZ".into(),
            version: None,
            source: "ZIP member-name hints".into(),
            notes: vec!["ZIP container with NumPy-like entries".into()],
            pattern_matches: matches.clone(),
        });
        out.data_structures.push(DataStructureHint {
            name: "ndarray_bundle".into(),
            source: "ZIP member-name hints".into(),
            offset: Some(base_offset),
            length: None,
            details: BTreeMap::new(),
            pattern_matches: matches,
        });
    }

    if out.specs.is_empty() {
        out.warnings.push(
            "ZIP container detected, but no LLM-specific member-name hints were confirmed".into(),
        );
    }

    extract_common_model_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "ZIP text region",
        &mut out,
    );
    extract_quantization_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "ZIP text region",
        &mut out,
    );
    extract_parameter_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "ZIP text region",
        &mut out,
    );
    extract_dataset_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "ZIP text region",
        &mut out,
    );
    out
}

fn detect_onnx(ctx: &SharedScanContext, base_offset: u64) -> DetectorOutput {
    let mut out = DetectorOutput::default();
    let onnx_hits = find_token_matches(
        &ctx.lower_scan,
        b"onnx",
        base_offset,
        DEFAULT_MAX_POSITIONS_PER_PATTERN,
        TokenBoundaryMode::AlphaNum,
    );
    let graph_hits = collect_key_matches(
        &ctx.lower_scan,
        &[
            b"graph",
            b"initializer",
            b"tensorproto",
            b"ir_version",
            b"opset",
        ],
        base_offset,
        DEFAULT_MAX_POSITIONS_PER_PATTERN,
    );
    let mut score = 0;
    if !onnx_hits.is_empty() {
        score += 3;
    }
    if !graph_hits.is_empty() {
        score += graph_hits.len().min(3) as i32;
    }
    if ctx.file_name_lower.ends_with(b".onnx") {
        score += 2;
    }
    if score < 4 {
        return out;
    }

    let mut matches = onnx_hits.clone();
    extend_pattern_matches_limited(
        &mut matches,
        graph_hits.clone(),
        DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
    );
    out.specs.push(SpecDetection {
        name: "ONNX".into(),
        version: None,
        source: "protobuf/graph text hints".into(),
        notes: vec!["graph/tensor metadata hints present".into()],
        pattern_matches: matches.clone(),
    });

    if graph_hits.len() >= 2 {
        out.data_structures.push(DataStructureHint {
            name: "computation_graph".into(),
            source: "ONNX graph hints".into(),
            offset: matches.first().map(|m| m.offset),
            length: None,
            details: BTreeMap::new(),
            pattern_matches: matches.clone(),
        });
    }

    extract_common_model_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "ONNX text region",
        &mut out,
    );
    extract_parameter_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "ONNX text region",
        &mut out,
    );
    out
}

fn detect_hdf5(scan_bytes: &[u8], ctx: &SharedScanContext, base_offset: u64) -> DetectorOutput {
    let mut out = DetectorOutput::default();
    if !has_prefix(scan_bytes, b"\x89HDF\r\n\x1a\n") {
        return out;
    }

    let magic = PatternMatch {
        pattern: "HDF5".into(),
        offset: base_offset,
        length: 8,
    };
    out.specs.push(SpecDetection {
        name: "HDF5".into(),
        version: None,
        source: "binary header".into(),
        notes: vec!["HDF5 magic bytes".into()],
        pattern_matches: vec![magic.clone()],
    });
    out.data_structures.push(DataStructureHint {
        name: "hierarchical_dataset".into(),
        source: "binary header".into(),
        offset: Some(base_offset),
        length: Some(8),
        details: BTreeMap::new(),
        pattern_matches: vec![magic],
    });

    extract_common_model_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "HDF5 text region",
        &mut out,
    );
    extract_parameter_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "HDF5 text region",
        &mut out,
    );
    extract_dataset_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "HDF5 text region",
        &mut out,
    );
    out
}

fn detect_json_structures(ctx: &SharedScanContext, base_offset: u64) -> DetectorOutput {
    let mut out = DetectorOutput::default();
    if !ctx.is_probably_text || !looks_like_json_structure(&ctx.lower_scan, ctx.jsonish_prefix) {
        return out;
    }

    let json_hits = collect_key_matches(&ctx.lower_scan, &[b"{", b":"], base_offset, 4);
    let model_matches = collect_key_matches(
        &ctx.lower_scan,
        &[b"\"model_type\"", b"\"architectures\""],
        base_offset,
        8,
    );
    let model_shape_matches = collect_key_matches(
        &ctx.lower_scan,
        &[
            b"\"hidden_size\"",
            b"\"num_attention_heads\"",
            b"\"num_hidden_layers\"",
        ],
        base_offset,
        8,
    );
    let tokenizer_matches = collect_key_matches(
        &ctx.lower_scan,
        &[
            b"\"tokenizer_class\"",
            b"\"added_tokens\"",
            b"\"vocab\"",
            b"\"merges\"",
        ],
        base_offset,
        12,
    );
    let dataset_matches = collect_key_matches(
        &ctx.lower_scan,
        &[
            b"\"dataset_info\"",
            b"\"splits\"",
            b"\"num_rows\"",
            b"\"download_size\"",
        ],
        base_offset,
        12,
    );

    if model_matches.is_empty()
        && tokenizer_matches.is_empty()
        && dataset_matches.is_empty()
        && ctx.structured_marker_count == 0
    {
        return out;
    }

    out.specs.push(SpecDetection {
        name: "JSON".into(),
        version: None,
        source: "text structure".into(),
        notes: vec!["JSON-like punctuation and LLM-related key/value layout".into()],
        pattern_matches: json_hits,
    });

    if !model_matches.is_empty() && !model_shape_matches.is_empty() {
        let mut matches = model_matches.clone();
        extend_pattern_matches_limited(
            &mut matches,
            model_shape_matches.clone(),
            DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
        );
        out.data_structures.push(DataStructureHint {
            name: "model_config".into(),
            source: "JSON keys".into(),
            offset: matches.first().map(|m| m.offset),
            length: None,
            details: BTreeMap::new(),
            pattern_matches: matches,
        });
    }

    if tokenizer_matches.len() >= 2 || contains_pattern(&ctx.lower_scan, b"\"tokenizer_class\"") {
        out.data_structures.push(DataStructureHint {
            name: "tokenizer_config_or_vocab".into(),
            source: "JSON keys".into(),
            offset: tokenizer_matches.first().map(|m| m.offset),
            length: None,
            details: BTreeMap::new(),
            pattern_matches: tokenizer_matches,
        });
    }

    if dataset_matches.len() >= 2 {
        out.data_structures.push(DataStructureHint {
            name: "dataset_metadata".into(),
            source: "JSON keys".into(),
            offset: dataset_matches.first().map(|m| m.offset),
            length: None,
            details: BTreeMap::new(),
            pattern_matches: dataset_matches,
        });
    }

    extract_common_model_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "JSON body",
        &mut out,
    );
    extract_quantization_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "JSON body",
        &mut out,
    );
    extract_parameter_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "JSON body",
        &mut out,
    );
    extract_dataset_hints(
        &ctx.lower_scan,
        base_offset,
        MatchDomain::Structured,
        "JSON body",
        &mut out,
    );
    out
}

fn detect_generic_hints(ctx: &SharedScanContext, base_offset: u64) -> DetectorOutput {
    let mut out = DetectorOutput::default();
    let has_transformer_shape = count_present_patterns(
        &ctx.lower_scan,
        &[
            b"hidden_size",
            b"num_hidden_layers",
            b"num_attention_heads",
            b"vocab_size",
        ],
    ) >= 2;
    let has_quant_context = contains_any_bytes(&ctx.lower_scan, QUANT_CONTEXT_KEYS);
    let has_dataset_context = contains_any_bytes(&ctx.lower_scan, DATASET_CONTEXT_KEYS);

    if !(ctx.is_probably_text
        || ctx.structured_marker_count >= 2
        || has_quant_context
        || has_dataset_context)
    {
        return out;
    }

    if has_transformer_shape {
        let matches = collect_key_matches(
            &ctx.lower_scan,
            &[
                b"hidden_size",
                b"num_hidden_layers",
                b"num_attention_heads",
                b"vocab_size",
            ],
            base_offset,
            DEFAULT_MAX_POSITIONS_PER_PATTERN,
        );
        out.data_structures.push(DataStructureHint {
            name: "transformer_config".into(),
            source: if ctx.is_probably_text {
                "generic text hints".into()
            } else {
                "binary string scan".into()
            },
            offset: matches.first().map(|m| m.offset),
            length: None,
            details: BTreeMap::new(),
            pattern_matches: matches,
        });
    }

    let source = if ctx.is_probably_text {
        "text"
    } else {
        "binary string scan"
    };
    if ctx.structured_marker_count >= 2 || has_transformer_shape {
        extract_common_model_hints(
            &ctx.lower_scan,
            base_offset,
            MatchDomain::Generic,
            source,
            &mut out,
        );
        extract_parameter_hints(
            &ctx.lower_scan,
            base_offset,
            MatchDomain::Generic,
            source,
            &mut out,
        );
    }
    if has_quant_context {
        extract_quantization_hints(
            &ctx.lower_scan,
            base_offset,
            MatchDomain::Generic,
            source,
            &mut out,
        );
    }
    if has_dataset_context {
        extract_dataset_hints(
            &ctx.lower_scan,
            base_offset,
            MatchDomain::Generic,
            source,
            &mut out,
        );
    }

    out
}

fn extract_common_model_hints(
    lower: &[u8],
    base_offset: u64,
    domain: MatchDomain,
    source: &str,
    out: &mut DetectorOutput,
) {
    const MODELS: &[(&[u8], &str, bool)] = &[
        (b"llama", "LLaMA", false),
        (b"mistral", "Mistral", false),
        (b"mixtral", "Mixtral", false),
        (b"qwen", "Qwen", false),
        (b"falcon", "Falcon", false),
        (b"gpt2", "GPT-2", false),
        (b"gptj", "GPT-J", false),
        (b"gpt-neox", "GPT-NeoX", false),
        (b"bert", "BERT", true),
        (b"roberta", "RoBERTa", false),
        (b"t5", "T5", true),
        (b"mpt", "MPT", true),
        (b"phi", "Phi", true),
        (b"gemma", "Gemma", false),
        (b"deepseek", "DeepSeek", false),
        (b"bloom", "BLOOM", false),
        (b"olmo", "OLMo", false),
        (b"granite", "Granite", false),
        (b"stablelm", "StableLM", false),
        (b"internlm", "InternLM", false),
        (b"baichuan", "Baichuan", false),
        (b"chatglm", "ChatGLM", false),
        (b"exaone", "EXAONE", false),
        (b"jamba", "Jamba", false),
        (b"starcoder", "StarCoder", false),
    ];

    let has_strong_model_context = contains_any_bytes(lower, STRONG_MODEL_CONTEXT_KEYS);
    if matches!(domain, MatchDomain::Generic) && !has_strong_model_context {
        return;
    }

    let mut ambiguous_matches = Vec::new();

    for (needle, family, ambiguous_token) in MODELS {
        let mut raw_matches = find_token_matches(
            lower,
            needle,
            base_offset,
            DEFAULT_MAX_POSITIONS_PER_PATTERN,
            TokenBoundaryMode::AlphaNum,
        );
        if raw_matches.is_empty() && !matches!(domain, MatchDomain::Generic) {
            raw_matches = find_key_aligned_value_matches(
                lower,
                needle,
                STRONG_MODEL_CONTEXT_KEYS,
                base_offset,
                DEFAULT_CONTEXT_RADIUS,
                DEFAULT_MAX_POSITIONS_PER_PATTERN,
            );
        }
        if raw_matches.is_empty() {
            continue;
        }

        let raw_backup = raw_matches.clone();
        let mut accepted = Vec::new();
        let mut score = 0i32;
        let mut saw_explicit_value = false;
        let mut saw_key_context = false;

        for item in raw_matches {
            let start = item.offset.saturating_sub(base_offset) as usize;
            let explicit_key_matches = collect_context_matches_near(
                lower,
                start,
                item.length as usize,
                base_offset,
                STRONG_MODEL_CONTEXT_KEYS,
                DEFAULT_CONTEXT_RADIUS,
                8,
            );
            let shape_matches = collect_context_matches_near(
                lower,
                start,
                item.length as usize,
                base_offset,
                MODEL_SHAPE_KEYS,
                DEFAULT_CONTEXT_RADIUS * 2,
                8,
            );
            let explicit_value = has_key_before_value_near(
                lower,
                start,
                item.length as usize,
                STRONG_MODEL_CONTEXT_KEYS,
                DEFAULT_CONTEXT_RADIUS,
            );
            let has_key_context = !explicit_key_matches.is_empty();
            let has_shape_context = !shape_matches.is_empty();
            let has_variant_context = has_variant_context_near(lower, needle, start);

            if explicit_value {
                score += 6;
                saw_explicit_value = true;
            } else if has_key_context {
                score += 3;
            }
            if has_key_context {
                saw_key_context = true;
            }
            if has_shape_context {
                score += 2;
            }
            if has_variant_context {
                score += 1;
            }

            if explicit_value || (has_key_context && has_shape_context && !*ambiguous_token) {
                accepted.push(item);
                extend_pattern_matches_limited(
                    &mut accepted,
                    explicit_key_matches,
                    DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
                );
                extend_pattern_matches_limited(
                    &mut accepted,
                    shape_matches,
                    DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
                );
            }
        }

        if !accepted.is_empty() {
            if raw_backup.len() > 1 {
                score += 1;
            }
            let threshold = if *ambiguous_token {
                6
            } else if matches!(domain, MatchDomain::Generic) {
                7
            } else {
                6
            };
            if score >= threshold && (saw_explicit_value || (!*ambiguous_token && saw_key_context))
            {
                let variant = extract_variant_hint(lower, needle, &accepted, base_offset);
                push_or_merge_model(
                    &mut out.models,
                    ModelHint {
                        family: (*family).to_string(),
                        variant,
                        source: source.to_string(),
                        pattern_matches: accepted,
                    },
                    DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
                );
                continue;
            }
        }

        extend_pattern_matches_limited(
            &mut ambiguous_matches,
            raw_backup,
            DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
        );
    }

    if !ambiguous_matches.is_empty() {
        push_or_merge_matched_pattern_group(
            &mut out.matched_patterns,
            MatchedPatternGroup {
                category: "model_family_token".into(),
                source: source.to_string(),
                pattern_matches: ambiguous_matches,
            },
            DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
        );
    }

    if contains_pattern(lower, b"model_type") {
        out.metadata
            .insert("contains_model_type".into(), "true".into());
    }
}

fn collect_context_matches_near(
    lower: &[u8],
    start: usize,
    length: usize,
    base_offset: u64,
    needles: &[&[u8]],
    radius: usize,
    max_total: usize,
) -> Vec<PatternMatch> {
    let window_start = start.saturating_sub(radius);
    let window_end = lower
        .len()
        .min(start.saturating_add(length).saturating_add(radius));
    let window = &lower[window_start..window_end];
    let mut out = Vec::new();
    for needle in needles {
        let remaining = max_total.saturating_sub(out.len());
        if remaining == 0 {
            break;
        }
        for pos in find_all_occurrences(window, needle, remaining) {
            out.push(PatternMatch {
                pattern: std::str::from_utf8(needle).unwrap_or_default().to_string(),
                offset: base_offset + (window_start + pos) as u64,
                length: needle.len() as u64,
            });
        }
    }
    dedup_pattern_matches(&mut out);
    if out.len() > max_total {
        out.truncate(max_total);
    }
    out
}

fn has_key_before_value_near(
    lower: &[u8],
    start: usize,
    length: usize,
    keys: &[&[u8]],
    max_gap: usize,
) -> bool {
    let window_start = start.saturating_sub(max_gap + 32);
    let window_end = lower
        .len()
        .min(start.saturating_add(length).saturating_add(2));
    let window = &lower[window_start..window_end];
    for key in keys {
        let positions = find_all_occurrences(window, key, 8);
        for pos in positions {
            let key_end = window_start + pos + key.len();
            if key_end <= start && start - key_end <= max_gap {
                return true;
            }
        }
    }
    false
}

fn has_variant_context_near(lower: &[u8], needle: &[u8], start: usize) -> bool {
    let end = lower.len().min(start + needle.len() + 32);
    let window = &lower[start..end];
    const SUFFIXES: &[&[u8]] = &[
        b"7b",
        b"8b",
        b"13b",
        b"14b",
        b"32b",
        b"70b",
        b"instruct",
        b"chat",
        b"base",
    ];
    SUFFIXES.iter().any(|suffix| {
        memmem(window, suffix)
            .map(|pos| is_edge_boundary(window, pos, suffix.len(), TokenBoundaryMode::AlphaNum))
            .unwrap_or(false)
    })
}

fn find_key_aligned_value_matches(
    lower: &[u8],
    needle: &[u8],
    keys: &[&[u8]],
    base_offset: u64,
    max_gap: usize,
    max_total: usize,
) -> Vec<PatternMatch> {
    let mut out = Vec::new();
    for key in keys {
        if out.len() >= max_total {
            break;
        }
        for key_pos in find_all_occurrences(lower, key, max_total) {
            let search_start = key_pos.saturating_add(key.len());
            let search_end = lower.len().min(search_start.saturating_add(max_gap));
            if search_start >= search_end {
                continue;
            }
            let window = &lower[search_start..search_end];
            let mut from = 0usize;
            while from + needle.len() <= window.len() && out.len() < max_total {
                let Some(rel) = memmem(&window[from..], needle) else {
                    break;
                };
                let pos = search_start + from + rel;
                out.push(PatternMatch {
                    pattern: std::str::from_utf8(needle).unwrap_or_default().to_string(),
                    offset: base_offset + pos as u64,
                    length: needle.len() as u64,
                });
                from += rel + needle.len();
            }
        }
    }
    dedup_pattern_matches(&mut out);
    if out.len() > max_total {
        out.truncate(max_total);
    }
    out
}

fn extract_variant_hint(
    lower: &[u8],
    needle: &[u8],
    matches: &[PatternMatch],
    base_offset: u64,
) -> Option<String> {
    let first = matches.first()?;
    let start = first.offset.saturating_sub(base_offset) as usize;
    let end = lower.len().min(start + needle.len() + 32);
    let window = &lower[start..end];
    const SUFFIXES: &[&[u8]] = &[
        b"7b",
        b"8b",
        b"13b",
        b"14b",
        b"32b",
        b"70b",
        b"instruct",
        b"chat",
        b"base",
    ];
    for suffix in SUFFIXES {
        if let Some(pos) = memmem(window, suffix) {
            let suffix_end = pos + suffix.len();
            if is_edge_boundary(window, pos, suffix_end - pos, TokenBoundaryMode::AlphaNum) {
                let base = std::str::from_utf8(needle).unwrap_or_default();
                let suffix_text = std::str::from_utf8(suffix).unwrap_or_default();
                return Some(format!("{} {}", base, suffix_text));
            }
        }
    }
    None
}

fn extract_quantization_hints(
    lower: &[u8],
    base_offset: u64,
    domain: MatchDomain,
    source: &str,
    out: &mut DetectorOutput,
) {
    const QUANTS: &[&[u8]] = &[
        b"q2_k",
        b"q3_k",
        b"q4_0",
        b"q4_1",
        b"q4_k",
        b"q5_0",
        b"q5_1",
        b"q5_k",
        b"q6_k",
        b"q8_0",
        b"int4",
        b"int8",
        b"fp4",
        b"fp8",
        b"nf4",
        b"gptq",
        b"awq",
        b"ggml",
        b"gguf",
        b"bitsandbytes",
    ];

    for needle in QUANTS {
        let raw_matches = find_token_matches(
            lower,
            needle,
            base_offset,
            DEFAULT_MAX_POSITIONS_PER_PATTERN,
            TokenBoundaryMode::AlphaNumUnderscore,
        );
        if raw_matches.is_empty() {
            continue;
        }

        let mut accepted = Vec::new();
        let mut score = 0i32;
        for item in raw_matches {
            let start = item.offset.saturating_sub(base_offset) as usize;
            let contextual = has_any_context_near(
                lower,
                start,
                item.length as usize,
                QUANT_CONTEXT_KEYS,
                DEFAULT_CONTEXT_RADIUS,
            );
            if contextual {
                score += 3;
                accepted.push(item);
            } else if matches!(domain, MatchDomain::FormatSpecific) {
                score += 1;
                accepted.push(item);
            }
        }

        let threshold = match domain {
            MatchDomain::FormatSpecific => 1,
            MatchDomain::Structured => 3,
            MatchDomain::Generic => 4,
        };
        if score < threshold || accepted.is_empty() {
            continue;
        }

        let scheme = std::str::from_utf8(needle).unwrap_or_default().to_string();
        push_or_merge_quantization(
            &mut out.quantization,
            QuantizationHint {
                scheme,
                source: source.to_string(),
                pattern_matches: accepted,
            },
            DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
        );
    }
}

fn extract_parameter_hints(
    lower: &[u8],
    base_offset: u64,
    domain: MatchDomain,
    source: &str,
    out: &mut DetectorOutput,
) {
    const PARAMETER_KEYS: &[&[u8]] = &[
        b"hidden_size",
        b"intermediate_size",
        b"num_hidden_layers",
        b"num_attention_heads",
        b"vocab_size",
        b"context_length",
        b"embedding_length",
        b"block_count",
    ];
    for key in PARAMETER_KEYS {
        for (value, matches) in find_jsonish_numeric_values(lower, key, base_offset, 4) {
            push_or_merge_parameter(
                &mut out.parameter_data,
                ParameterHint {
                    metric: std::str::from_utf8(key).unwrap_or_default().to_string(),
                    value,
                    source: source.to_string(),
                    pattern_matches: matches,
                },
                DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
            );
        }
    }

    let suffix_matches = find_numeric_suffix_values(
        lower,
        &[b"b parameters", b"m parameters", b"b params", b"m params"],
        &[b"parameters", b"params", b"model", b"weights"],
        base_offset,
        matches!(domain, MatchDomain::FormatSpecific),
        8,
    );
    for (value, matches) in suffix_matches {
        push_or_merge_parameter(
            &mut out.parameter_data,
            ParameterHint {
                metric: "declared_parameter_count".into(),
                value,
                source: source.to_string(),
                pattern_matches: matches,
            },
            DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
        );
    }
}

fn extract_dataset_hints(
    lower: &[u8],
    base_offset: u64,
    domain: MatchDomain,
    source: &str,
    out: &mut DetectorOutput,
) {
    const DATASET_KEYS: &[&[u8]] = &[
        b"num_rows",
        b"download_size",
        b"dataset_size",
        b"num_examples",
        b"train_size",
    ];
    for key in DATASET_KEYS {
        for (value, matches) in find_jsonish_numeric_values(lower, key, base_offset, 4) {
            push_or_merge_dataset(
                &mut out.dataset_size,
                DatasetSizeHint {
                    metric: std::str::from_utf8(key).unwrap_or_default().to_string(),
                    value,
                    source: source.to_string(),
                    pattern_matches: matches,
                },
                DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
            );
        }
    }

    let suffix_matches = find_numeric_suffix_values(
        lower,
        &[b"tokens", b"samples", b"examples", b"rows"],
        DATASET_CONTEXT_KEYS,
        base_offset,
        matches!(domain, MatchDomain::FormatSpecific)
            && contains_any_bytes(lower, DATASET_CONTEXT_KEYS),
        8,
    );
    for (value, matches) in suffix_matches {
        push_or_merge_dataset(
            &mut out.dataset_size,
            DatasetSizeHint {
                metric: "declared_dataset_measure".into(),
                value,
                source: source.to_string(),
                pattern_matches: matches,
            },
            DEFAULT_MAX_PATTERN_MATCHES_PER_ITEM,
        );
    }
}

fn scan_embedded_headers(
    window: &[u8],
    base_offset: u64,
    analysis: &mut Analysis,
    options: &AnalysisOptions,
) {
    let magic_sets: &[(&[u8], &str)] = &[
        (b"GGUF", "GGUF"),
        (b"PK\x03\x04", "ZIP"),
        (b"\x89HDF\r\n\x1a\n", "HDF5"),
    ];

    for (magic, label) in magic_sets {
        for offset in find_all_occurrences(window, magic, 12) {
            let absolute_offset = base_offset + offset as u64;
            if absolute_offset == 0 {
                continue;
            }
            let pattern_match = PatternMatch {
                pattern: (*label).to_string(),
                offset: absolute_offset,
                length: magic.len() as u64,
            };
            let mut details = BTreeMap::new();
            details.insert("embedded_magic".into(), (*label).to_string());

            push_limited_data_structure(
                analysis,
                DataStructureHint {
                    name: format!("embedded_{}_header", label.to_ascii_lowercase()),
                    source: "deep header scan".into(),
                    offset: Some(absolute_offset),
                    length: Some(magic.len() as u64),
                    details: details.clone(),
                    pattern_matches: vec![pattern_match.clone()],
                },
                options,
            );
            push_limited_shaper(
                analysis,
                EntroshapeHint {
                    kind: "embedded_header".into(),
                    offset: absolute_offset,
                    length: Some(magic.len() as u64),
                    description: format!("embedded {} header found away from file start", label),
                    source: "deep header scan".into(),
                    details,
                    pattern_matches: vec![pattern_match],
                },
                options,
            );
        }
    }
}

fn maybe_start_streaming_safetensors_header(
    state: &mut DeepScanState,
    chunk: &[u8],
    chunk_offset: u64,
    options: &AnalysisOptions,
    analysis: &mut Analysis,
) {
    if chunk_offset != 0 || state.root_safetensors.is_some() || chunk.len() < 9 {
        return;
    }

    let Some(header_len) = read_le_u64(&chunk[..8]) else {
        return;
    };
    if header_len == 0 || chunk[8] != b'{' {
        return;
    }

    analysis
        .metadata
        .insert("safetensors_header_bytes".into(), header_len.to_string());
    analysis
        .metadata
        .insert("safetensors_streaming_parse".into(), "true".into());
    analysis.metadata.insert(
        "safetensors_root_parse_strategy".into(),
        "streamed_full_header".into(),
    );
    if header_len > options.max_safetensors_header_bytes as u64 {
        analysis.metadata.insert(
            "safetensors_prefix_limit_overridden_in_deep_mode".into(),
            "true".into(),
        );
    }
    state.root_safetensors = Some(StreamingSafetensorsHeaderState::new(header_len));
}

fn stream_safetensors_header_keywords(
    state: &mut DeepScanState,
    chunk: &[u8],
    chunk_offset: u64,
    analysis: &mut Analysis,
    options: &AnalysisOptions,
) {
    let Some(mut stream) = state.root_safetensors.take() else {
        return;
    };

    let header_start = 8u64;
    let header_end = header_start.saturating_add(stream.declared_header_bytes);
    let chunk_end = chunk_offset.saturating_add(chunk.len() as u64);
    if chunk_end <= header_start || chunk_offset >= header_end {
        state.root_safetensors = Some(stream);
        return;
    }

    let local_start = header_start.saturating_sub(chunk_offset) as usize;
    let local_end = (header_end.min(chunk_end) - chunk_offset) as usize;
    let slice = &chunk[local_start..local_end];
    if slice.is_empty() {
        state.root_safetensors = Some(stream);
        return;
    }

    if stream.processed_header_bytes == 0 {
        stream.saw_open_brace = slice.first() == Some(&b'{');
    }
    if chunk_offset + local_end as u64 == header_end {
        stream.saw_close_brace = slice.last() == Some(&b'}');
    }

    let lower = ascii_lower_vec(slice);
    let mut combined = Vec::with_capacity(stream.keyword_tail.len() + lower.len());
    combined.extend_from_slice(&stream.keyword_tail);
    combined.extend_from_slice(&lower);
    let combined_base_offset =
        (chunk_offset + local_start as u64).saturating_sub(stream.keyword_tail.len() as u64);

    if stream.metadata_pos.is_none() {
        if let Some(pos) = memmem(&combined, b"\"__metadata__\"") {
            stream.metadata_pos = Some(combined_base_offset + pos as u64);
        }
    }
    if stream.dtype_pos.is_none() {
        if let Some(pos) = memmem(&combined, b"\"dtype\"") {
            stream.dtype_pos = Some(combined_base_offset + pos as u64);
        }
    }
    if stream.data_offsets_pos.is_none() {
        if let Some(pos) = memmem(&combined, b"\"data_offsets\"") {
            stream.data_offsets_pos = Some(combined_base_offset + pos as u64);
        }
    }

    let keep = STREAMING_KEYWORD_TAIL.min(combined.len());
    stream.keyword_tail.clear();
    stream
        .keyword_tail
        .extend_from_slice(&combined[combined.len() - keep..]);
    stream.processed_header_bytes = stream
        .processed_header_bytes
        .saturating_add(slice.len() as u64);

    if stream.processed_header_bytes < stream.declared_header_bytes {
        state.root_safetensors = Some(stream);
        return;
    }

    let metadata_pos = stream.metadata_pos;
    let dtype_pos = stream.dtype_pos;
    let data_offsets_pos = stream.data_offsets_pos;
    let header_len = stream.declared_header_bytes;
    let brace_ok = stream.saw_open_brace && stream.saw_close_brace;

    let mut pattern_matches = Vec::new();
    if let Some(pos) = metadata_pos {
        pattern_matches.push(PatternMatch {
            pattern: "\"__metadata__\"".into(),
            offset: pos,
            length: 14,
        });
    }
    if let Some(pos) = dtype_pos {
        pattern_matches.push(PatternMatch {
            pattern: "\"dtype\"".into(),
            offset: pos,
            length: 7,
        });
    }
    if let Some(pos) = data_offsets_pos {
        pattern_matches.push(PatternMatch {
            pattern: "\"data_offsets\"".into(),
            offset: pos,
            length: 14,
        });
    }

    if !brace_ok {
        push_unique_string(
            &mut analysis.warnings,
            "safetensors streamed header did not preserve opening/closing JSON braces".into(),
        );
        return;
    }

    let mut score = 2;
    if metadata_pos.is_some() {
        score += 3;
    }
    if dtype_pos.is_some() {
        score += 2;
    }
    if data_offsets_pos.is_some() {
        score += 2;
    }
    if score < 5 {
        return;
    }

    push_or_merge_spec(
        &mut analysis.detected_specs,
        SpecDetection {
            name: "safetensors".into(),
            version: None,
            source: "streamed root header".into(),
            notes: vec![format!("header_bytes={}", header_len)],
            pattern_matches: pattern_matches.clone(),
        },
        options.max_pattern_matches_per_item,
    );

    let mut details = BTreeMap::new();
    details.insert("header_bytes".into(), header_len.to_string());
    push_limited_data_structure(
        analysis,
        DataStructureHint {
            name: "tensor_index".into(),
            source: "streamed root header".into(),
            offset: Some(8),
            length: Some(header_len),
            details,
            pattern_matches,
        },
        options,
    );
}

fn scan_entropy_windows(
    chunk: &[u8],
    chunk_offset: u64,
    state: &mut DeepScanState,
    analysis: &mut Analysis,
    options: &AnalysisOptions,
) {
    let window_size = options.deep_entropy_window_bytes.max(1024);
    let mut start = 0usize;
    while start < chunk.len() {
        let end = chunk.len().min(start + window_size);
        let slice = &chunk[start..end];
        let entropy = estimate_entropy_sample(slice, slice.len());

        if entropy >= HIGH_ENTROPY_THRESHOLD {
            let mut details = BTreeMap::new();
            details.insert("entropy".into(), format!("{:.6}", entropy));
            push_limited_shaper(
                analysis,
                EntroshapeHint {
                    kind: "high_entropy_region".into(),
                    offset: chunk_offset + start as u64,
                    length: Some(slice.len() as u64),
                    description: format!("entropy {:.3} exceeded high-entropy threshold", entropy),
                    source: "deep entropy scan".into(),
                    details,
                    pattern_matches: Vec::new(),
                },
                options,
            );
        } else if entropy <= LOW_ENTROPY_THRESHOLD {
            let mut details = BTreeMap::new();
            details.insert("entropy".into(), format!("{:.6}", entropy));
            push_limited_shaper(
                analysis,
                EntroshapeHint {
                    kind: "low_entropy_region".into(),
                    offset: chunk_offset + start as u64,
                    length: Some(slice.len() as u64),
                    description: format!("entropy {:.3} fell below low-entropy threshold", entropy),
                    source: "deep entropy scan".into(),
                    details,
                    pattern_matches: Vec::new(),
                },
                options,
            );
        }

        if let Some(previous) = state.previous_entropy {
            let delta = (entropy - previous).abs();
            if delta >= ENTROPY_TRANSITION_DELTA {
                let mut details = BTreeMap::new();
                details.insert("previous_entropy".into(), format!("{:.6}", previous));
                details.insert("current_entropy".into(), format!("{:.6}", entropy));
                details.insert("delta".into(), format!("{:.6}", delta));
                push_limited_shaper(
                    analysis,
                    EntroshapeHint {
                        kind: "entropy_transition".into(),
                        offset: chunk_offset + start as u64,
                        length: Some(slice.len() as u64),
                        description: format!(
                            "entropy changed by {:.3} between adjacent windows",
                            delta
                        ),
                        source: "deep entropy scan".into(),
                        details,
                        pattern_matches: Vec::new(),
                    },
                    options,
                );
            }
        }

        state.previous_entropy = Some(entropy);
        start = end;
    }
}

fn scan_zero_runs(
    chunk: &[u8],
    chunk_offset: u64,
    state: &mut DeepScanState,
    analysis: &mut Analysis,
    options: &AnalysisOptions,
) {
    for (index, byte) in chunk.iter().copied().enumerate() {
        let absolute_offset = chunk_offset + index as u64;
        if byte == 0 {
            if state.zero_run_start.is_none() {
                state.zero_run_start = Some(absolute_offset);
                state.zero_run_length = 0;
            }
            state.zero_run_length = state.zero_run_length.saturating_add(1);
        } else if state.zero_run_start.is_some() {
            if state.zero_run_length >= ZERO_RUN_ANOMALY_THRESHOLD {
                let mut details = BTreeMap::new();
                details.insert("zero_run_bytes".into(), state.zero_run_length.to_string());
                push_limited_shaper(
                    analysis,
                    EntroshapeHint {
                        kind: "long_zero_run".into(),
                        offset: state.zero_run_start.unwrap_or(absolute_offset),
                        length: Some(state.zero_run_length),
                        description: format!("zero-byte run of {} bytes", state.zero_run_length),
                        source: "deep zero-run scan".into(),
                        details,
                        pattern_matches: Vec::new(),
                    },
                    options,
                );
            }
            state.zero_run_start = None;
            state.zero_run_length = 0;
        }
    }
}

fn find_jsonish_numeric_values(
    lower: &[u8],
    key: &[u8],
    base_offset: u64,
    max_results: usize,
) -> Vec<(String, Vec<PatternMatch>)> {
    let mut out = Vec::new();
    let mut search_from = 0usize;
    while search_from + key.len() <= lower.len() && out.len() < max_results {
        let Some(rel) = memmem(&lower[search_from..], key) else {
            break;
        };
        let pos = search_from + rel;
        if !is_edge_boundary(lower, pos, key.len(), TokenBoundaryMode::AlphaNumUnderscore) {
            search_from = pos + 1;
            continue;
        }

        let mut cursor = pos + key.len();
        while cursor < lower.len() && lower[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor >= lower.len() {
            break;
        }

        if lower[cursor] == b'"' || lower[cursor] == b'\'' {
            cursor += 1;
            while cursor < lower.len() && lower[cursor] != b'"' && lower[cursor] != b'\'' {
                cursor += 1;
            }
            cursor += 1;
            while cursor < lower.len() && lower[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
        }

        if cursor >= lower.len() || !(lower[cursor] == b':' || lower[cursor] == b'=') {
            search_from = pos + 1;
            continue;
        }
        cursor += 1;
        while cursor < lower.len() && lower[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor < lower.len() && (lower[cursor] == b'"' || lower[cursor] == b'\'') {
            cursor += 1;
        }

        let Some((value_end, value)) = take_numberish_span(lower, cursor) else {
            search_from = pos + 1;
            continue;
        };

        let matches = vec![
            PatternMatch {
                pattern: std::str::from_utf8(key).unwrap_or_default().to_string(),
                offset: base_offset + pos as u64,
                length: key.len() as u64,
            },
            PatternMatch {
                pattern: value.clone(),
                offset: base_offset + cursor as u64,
                length: (value_end - cursor) as u64,
            },
        ];
        out.push((value, matches));
        search_from = value_end;
    }
    out
}

fn find_numeric_suffix_values(
    lower: &[u8],
    suffixes: &[&[u8]],
    context_keywords: &[&[u8]],
    base_offset: u64,
    allow_without_context: bool,
    max_results: usize,
) -> Vec<(String, Vec<PatternMatch>)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < lower.len() && out.len() < max_results {
        if !lower[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        if i > 0 && lower[i - 1].is_ascii_alphanumeric() {
            i += 1;
            continue;
        }

        let Some((number_end, number)) = take_numberish_span(lower, i) else {
            i += 1;
            continue;
        };
        let mut suffix_cursor = number_end;
        while suffix_cursor < lower.len() && lower[suffix_cursor].is_ascii_whitespace() {
            suffix_cursor += 1;
        }

        let mut matched_suffix: Option<&[u8]> = None;
        for suffix in suffixes {
            let end = suffix_cursor.saturating_add(suffix.len());
            if end <= lower.len() && &lower[suffix_cursor..end] == *suffix {
                matched_suffix = Some(*suffix);
                break;
            }
        }

        let Some(suffix) = matched_suffix else {
            i = number_end;
            continue;
        };

        let with_context = has_any_context_near(
            lower,
            i,
            suffix_cursor + suffix.len() - i,
            context_keywords,
            DEFAULT_CONTEXT_RADIUS,
        );
        if !with_context && !allow_without_context {
            i = number_end;
            continue;
        }

        let suffix_text = std::str::from_utf8(suffix).unwrap_or_default();
        let value = format!("{} {}", number, suffix_text);
        out.push((
            value,
            vec![
                PatternMatch {
                    pattern: number.clone(),
                    offset: base_offset + i as u64,
                    length: (number_end - i) as u64,
                },
                PatternMatch {
                    pattern: suffix_text.to_string(),
                    offset: base_offset + suffix_cursor as u64,
                    length: suffix.len() as u64,
                },
            ],
        ));
        i = suffix_cursor + suffix.len();
    }
    out
}

fn take_numberish_span(lower: &[u8], start: usize) -> Option<(usize, String)> {
    if start >= lower.len() {
        return None;
    }
    let mut end = start;
    let mut started = false;
    while end < lower.len() {
        let b = lower[end];
        if b.is_ascii_digit() || matches!(b, b'.' | b',' | b'_' | b'-' | b'+' | b'e' | b'E') {
            started = true;
            end += 1;
        } else {
            break;
        }
    }
    if !started {
        return None;
    }
    let raw = std::str::from_utf8(&lower[start..end])
        .ok()?
        .trim_matches(',')
        .to_string();
    if raw.is_empty() {
        None
    } else {
        Some((end, raw))
    }
}

fn collect_key_matches(
    lower: &[u8],
    needles: &[&[u8]],
    base_offset: u64,
    max_total: usize,
) -> Vec<PatternMatch> {
    let mut out = Vec::new();
    for needle in needles {
        let limit = max_total.saturating_sub(out.len());
        if limit == 0 {
            break;
        }
        let positions = find_all_occurrences(lower, needle, limit);
        for pos in positions {
            out.push(PatternMatch {
                pattern: std::str::from_utf8(needle).unwrap_or_default().to_string(),
                offset: base_offset + pos as u64,
                length: needle.len() as u64,
            });
        }
    }
    dedup_pattern_matches(&mut out);
    out
}

fn find_token_matches(
    lower: &[u8],
    needle: &[u8],
    base_offset: u64,
    max_total: usize,
    mode: TokenBoundaryMode,
) -> Vec<PatternMatch> {
    let mut out = Vec::new();
    let mut search_from = 0usize;
    while search_from + needle.len() <= lower.len() && out.len() < max_total {
        let Some(rel) = memmem(&lower[search_from..], needle) else {
            break;
        };
        let pos = search_from + rel;
        if is_edge_boundary(lower, pos, needle.len(), mode) {
            out.push(PatternMatch {
                pattern: std::str::from_utf8(needle).unwrap_or_default().to_string(),
                offset: base_offset + pos as u64,
                length: needle.len() as u64,
            });
        }
        search_from = pos + 1;
    }
    out
}

fn find_all_occurrences(haystack: &[u8], needle: &[u8], max_total: usize) -> Vec<usize> {
    let mut out = Vec::new();
    if needle.is_empty() {
        return out;
    }
    let mut search_from = 0usize;
    while search_from + needle.len() <= haystack.len() && out.len() < max_total {
        let Some(rel) = memmem(&haystack[search_from..], needle) else {
            break;
        };
        let pos = search_from + rel;
        out.push(pos);
        search_from = pos + 1;
    }
    out
}

fn has_any_context_near(
    lower: &[u8],
    start: usize,
    length: usize,
    context_keywords: &[&[u8]],
    radius: usize,
) -> bool {
    let window_start = start.saturating_sub(radius);
    let window_end = lower
        .len()
        .min(start.saturating_add(length).saturating_add(radius));
    contains_any_bytes(&lower[window_start..window_end], context_keywords)
}

fn count_present_patterns(haystack: &[u8], needles: &[&[u8]]) -> usize {
    needles
        .iter()
        .filter(|needle| memmem(haystack, needle).is_some())
        .count()
}

fn contains_any_bytes(haystack: &[u8], needles: &[&[u8]]) -> bool {
    needles
        .iter()
        .any(|needle| memmem(haystack, needle).is_some())
}

fn contains_pattern(haystack: &[u8], needle: &[u8]) -> bool {
    memmem(haystack, needle).is_some()
}

fn looks_like_json_structure(lower: &[u8], jsonish_prefix: bool) -> bool {
    if !jsonish_prefix {
        return false;
    }
    let sample = &lower[..lower.len().min(4096)];
    let quotes = count_byte(sample, b'"');
    let colons = count_byte(sample, b':');
    let openings = count_byte(sample, b'{') + count_byte(sample, b'[');
    let closings = count_byte(sample, b'}') + count_byte(sample, b']');
    quotes >= 2 && colons >= 1 && openings >= 1 && closings >= 1
}

fn count_byte(bytes: &[u8], needle: u8) -> usize {
    bytes.iter().filter(|&&b| b == needle).count()
}

fn is_likely_textual_or_mixed(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let sample = &bytes[..bytes.len().min(TEXT_SAMPLE_MAX)];
    let mut printable = 0usize;
    let mut punctuation = 0usize;
    for &b in sample {
        if b == b'\n' || b == b'\r' || b == b'\t' || (0x20..=0x7e).contains(&b) {
            printable += 1;
        }
        if matches!(b, b'"' | b':' | b'{' | b'}' | b'[' | b']' | b'_' | b'-') {
            punctuation += 1;
        }
    }
    let printable_ratio = printable as f64 / sample.len() as f64;
    printable_ratio > 0.20 || punctuation >= 4
}

fn is_jsonish_text_prefix(bytes: &[u8]) -> bool {
    matches!(first_non_whitespace_byte(bytes), Some(b'{') | Some(b'['))
}

fn first_non_whitespace_byte(bytes: &[u8]) -> Option<u8> {
    bytes.iter().find(|&&b| !b.is_ascii_whitespace()).copied()
}

fn is_edge_boundary(haystack: &[u8], start: usize, len: usize, mode: TokenBoundaryMode) -> bool {
    let end = start.saturating_add(len);
    let left_ok = if start == 0 {
        true
    } else {
        !is_word_byte(haystack[start - 1], mode)
    };
    let right_ok = if end >= haystack.len() {
        true
    } else {
        !is_word_byte(haystack[end], mode)
    };
    left_ok && right_ok
}

fn is_word_byte(byte: u8, mode: TokenBoundaryMode) -> bool {
    match mode {
        TokenBoundaryMode::AlphaNum => byte.is_ascii_alphanumeric(),
        TokenBoundaryMode::AlphaNumUnderscore => byte.is_ascii_alphanumeric() || byte == b'_',
    }
}

fn ascii_lower_vec(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    out.extend(bytes.iter().map(|b| b.to_ascii_lowercase()));
    out
}

fn memmem(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if haystack.len() < needle.len() {
        return None;
    }

    let first = needle[0];
    let last_start = haystack.len() - needle.len();
    let mut i = 0usize;
    while i <= last_start {
        if haystack[i] == first && &haystack[i..i + needle.len()] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn has_prefix(bytes: &[u8], prefix: &[u8]) -> bool {
    bytes.len() >= prefix.len() && &bytes[..prefix.len()] == prefix
}

fn read_le_u32(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 4 {
        return None;
    }
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_le_u64(bytes: &[u8]) -> Option<u64> {
    if bytes.len() < 8 {
        return None;
    }
    Some(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn is_probably_text(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    let sample = &bytes[..bytes.len().min(TEXT_SAMPLE_MAX)];
    let mut textish = 0usize;
    for &b in sample {
        if b == b'\n' || b == b'\r' || b == b'\t' || (0x20..=0x7e).contains(&b) {
            textish += 1;
        }
    }
    (textish as f64 / sample.len() as f64) > 0.85
}

fn estimate_entropy_sample(bytes: &[u8], max_len: usize) -> f64 {
    let sample = &bytes[..bytes.len().min(max_len)];
    if sample.is_empty() {
        return 0.0;
    }

    let mut counts = [0usize; 256];
    for &b in sample {
        counts[b as usize] += 1;
    }
    let len = sample.len() as f64;
    let mut entropy = 0.0;
    for &count in &counts {
        if count == 0 {
            continue;
        }
        let p = count as f64 / len;
        entropy -= p * p.log2();
    }
    entropy
}

fn should_parallelize(byte_count: u64) -> bool {
    byte_count >= PARALLEL_THRESHOLD_BYTES
        && std::thread::available_parallelism()
            .map(|n| n.get() > 1)
            .unwrap_or(false)
}

fn finalize_analysis(analysis: &mut Analysis) {
    analysis.signatures.sort();
    analysis.signatures.dedup();
    analysis
        .detected_specs
        .sort_by(|a, b| a.name.cmp(&b.name).then(a.version.cmp(&b.version)));
    analysis
        .detected_models
        .sort_by(|a, b| a.family.cmp(&b.family).then(a.variant.cmp(&b.variant)));
    analysis
        .matched_patterns
        .sort_by(|a, b| a.category.cmp(&b.category).then(a.source.cmp(&b.source)));
    analysis.detected_data_structures.sort_by(|a, b| {
        a.offset
            .cmp(&b.offset)
            .then(a.name.cmp(&b.name))
            .then(a.source.cmp(&b.source))
    });
    analysis
        .quantization
        .sort_by(|a, b| a.scheme.cmp(&b.scheme));
    analysis
        .dataset_size
        .sort_by(|a, b| a.metric.cmp(&b.metric).then(a.value.cmp(&b.value)));
    analysis
        .parameter_data
        .sort_by(|a, b| a.metric.cmp(&b.metric).then(a.value.cmp(&b.value)));
    analysis
        .shapes
        .sort_by(|a, b| a.offset.cmp(&b.offset).then(a.kind.cmp(&b.kind)));
    analysis.warnings.sort();
    analysis.warnings.dedup();
}

fn push_or_merge_spec(vec: &mut Vec<SpecDetection>, mut item: SpecDetection, max_matches: usize) {
    trim_pattern_matches(&mut item.pattern_matches, max_matches);
    if let Some(existing) = vec
        .iter_mut()
        .find(|x| x.name == item.name && x.version == item.version)
    {
        merge_string_vectors(&mut existing.notes, item.notes);
        merge_pattern_matches(
            &mut existing.pattern_matches,
            item.pattern_matches,
            max_matches,
        );
        if existing.source.is_empty() {
            existing.source = item.source;
        }
    } else {
        vec.push(item);
    }
}

fn push_or_merge_model(vec: &mut Vec<ModelHint>, mut item: ModelHint, max_matches: usize) {
    trim_pattern_matches(&mut item.pattern_matches, max_matches);
    if let Some(existing) = vec
        .iter_mut()
        .find(|x| x.family == item.family && x.variant == item.variant)
    {
        merge_pattern_matches(
            &mut existing.pattern_matches,
            item.pattern_matches,
            max_matches,
        );
        if existing.source.is_empty() {
            existing.source = item.source;
        }
    } else {
        vec.push(item);
    }
}

fn push_or_merge_matched_pattern_group(
    vec: &mut Vec<MatchedPatternGroup>,
    mut item: MatchedPatternGroup,
    max_matches: usize,
) {
    trim_pattern_matches(&mut item.pattern_matches, max_matches);
    if item.pattern_matches.is_empty() {
        return;
    }
    if let Some(existing) = vec
        .iter_mut()
        .find(|x| x.category == item.category && x.source == item.source)
    {
        merge_pattern_matches(
            &mut existing.pattern_matches,
            item.pattern_matches,
            max_matches,
        );
    } else {
        vec.push(item);
    }
}

fn push_or_merge_quantization(
    vec: &mut Vec<QuantizationHint>,
    mut item: QuantizationHint,
    max_matches: usize,
) {
    trim_pattern_matches(&mut item.pattern_matches, max_matches);
    if let Some(existing) = vec.iter_mut().find(|x| x.scheme == item.scheme) {
        merge_pattern_matches(
            &mut existing.pattern_matches,
            item.pattern_matches,
            max_matches,
        );
        if existing.source.is_empty() {
            existing.source = item.source;
        }
    } else {
        vec.push(item);
    }
}

fn push_or_merge_dataset(
    vec: &mut Vec<DatasetSizeHint>,
    mut item: DatasetSizeHint,
    max_matches: usize,
) {
    trim_pattern_matches(&mut item.pattern_matches, max_matches);
    if let Some(existing) = vec
        .iter_mut()
        .find(|x| x.metric == item.metric && x.value == item.value)
    {
        merge_pattern_matches(
            &mut existing.pattern_matches,
            item.pattern_matches,
            max_matches,
        );
        if existing.source.is_empty() {
            existing.source = item.source;
        }
    } else {
        vec.push(item);
    }
}

fn push_or_merge_parameter(
    vec: &mut Vec<ParameterHint>,
    mut item: ParameterHint,
    max_matches: usize,
) {
    trim_pattern_matches(&mut item.pattern_matches, max_matches);
    if let Some(existing) = vec
        .iter_mut()
        .find(|x| x.metric == item.metric && x.value == item.value)
    {
        merge_pattern_matches(
            &mut existing.pattern_matches,
            item.pattern_matches,
            max_matches,
        );
        if existing.source.is_empty() {
            existing.source = item.source;
        }
    } else {
        vec.push(item);
    }
}

fn push_limited_data_structure(
    analysis: &mut Analysis,
    mut item: DataStructureHint,
    options: &AnalysisOptions,
) {
    trim_pattern_matches(
        &mut item.pattern_matches,
        options.max_pattern_matches_per_item,
    );
    if let Some(existing) = analysis.detected_data_structures.iter_mut().find(|x| {
        x.name == item.name
            && x.offset == item.offset
            && x.length == item.length
            && x.source == item.source
    }) {
        merge_pattern_matches(
            &mut existing.pattern_matches,
            item.pattern_matches,
            options.max_pattern_matches_per_item,
        );
        merge_metadata_map(&mut existing.details, item.details);
        return;
    }

    if analysis.detected_data_structures.len() >= options.max_reported_structure_occurrences {
        increment_metadata_count(&mut analysis.metadata, "found_structure_occurrences");
        return;
    }
    analysis.detected_data_structures.push(item);
}

fn push_limited_shaper(analysis: &mut Analysis, mut item: EntroshapeHint, options: &AnalysisOptions) {
    trim_pattern_matches(
        &mut item.pattern_matches,
        options.max_pattern_matches_per_item,
    );
    if let Some(existing) = analysis.shapes.iter_mut().find(|x| {
        x.kind == item.kind
            && x.offset == item.offset
            && x.length == item.length
            && x.description == item.description
    }) {
        merge_pattern_matches(
            &mut existing.pattern_matches,
            item.pattern_matches,
            options.max_pattern_matches_per_item,
        );
        merge_metadata_map(&mut existing.details, item.details);
        return;
    }

    if analysis.shapes.len() >= options.max_reported_shapes {
        increment_metadata_count(&mut analysis.metadata, "found_shapes");
        return;
    }
    analysis.shapes.push(item);
}

fn merge_string_vectors(dst: &mut Vec<String>, src: Vec<String>) {
    for item in src {
        if !dst.iter().any(|x| x == &item) {
            dst.push(item);
        }
    }
}

fn merge_metadata_map(dst: &mut BTreeMap<String, String>, src: BTreeMap<String, String>) {
    for (k, v) in src {
        dst.entry(k).or_insert(v);
    }
}

fn merge_pattern_matches(dst: &mut Vec<PatternMatch>, src: Vec<PatternMatch>, max_matches: usize) {
    extend_pattern_matches_limited(dst, src, max_matches);
}

fn extend_pattern_matches_limited(
    dst: &mut Vec<PatternMatch>,
    src: Vec<PatternMatch>,
    max_matches: usize,
) {
    for item in src {
        if dst.iter().any(|x| {
            x.pattern == item.pattern && x.offset == item.offset && x.length == item.length
        }) {
            continue;
        }
        if dst.len() >= max_matches {
            break;
        }
        dst.push(item);
    }
}

fn trim_pattern_matches(matches: &mut Vec<PatternMatch>, max_matches: usize) {
    dedup_pattern_matches(matches);
    if matches.len() > max_matches {
        matches.truncate(max_matches);
    }
}

fn dedup_pattern_matches(matches: &mut Vec<PatternMatch>) {
    matches.sort_by(|a, b| {
        a.offset
            .cmp(&b.offset)
            .then(a.pattern.cmp(&b.pattern))
            .then(a.length.cmp(&b.length))
    });
    matches.dedup_by(|a, b| a.pattern == b.pattern && a.offset == b.offset && a.length == b.length);
}

fn increment_metadata_count(metadata: &mut BTreeMap<String, String>, key: &str) {
    let current = metadata
        .get(key)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    metadata.insert(key.to_string(), current.saturating_add(1).to_string());
}

fn push_unique_string(vec: &mut Vec<String>, item: String) {
    if !vec.iter().any(|x| x == &item) {
        vec.push(item);
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

fn now_utc_iso() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn error_to_json(kind: &str, message: &str, file_name: &str, pretty: bool) -> String {
    let timestamp = now_utc_iso();
    let mut s = String::with_capacity(256 + message.len() + file_name.len());
    s.push('{');
    push_json_field_bool(&mut s, "ok", false, true);
    push_json_field_str(&mut s, "file_name", file_name, true);
    push_json_field_str(&mut s, "created_at_utc", &timestamp, true);
    s.push_str("\"error\":{");
    push_json_field_str(&mut s, "kind", kind, true);
    push_json_field_str(&mut s, "message", message, false);
    s.push_str("}}");
    maybe_pretty_json(s, pretty)
}

fn analysis_to_json(a: &Analysis, pretty: bool) -> String {
    let mut s = String::with_capacity(2048 + a.signatures.len() * 16 + a.warnings.len() * 32);
    s.push('{');
    push_json_field_bool(&mut s, "ok", a.ok, true);
    push_json_field_str(&mut s, "file_name", &a.file_name, true);
    push_json_field_str(&mut s, "created_at_utc", &a.created_at_utc, true);
    push_json_field_num(&mut s, "byte_count", a.byte_count, true);
    push_json_field_num(&mut s, "scanned_byte_count", a.scanned_byte_count, true);
    push_json_field_bool(&mut s, "is_probably_text", a.is_probably_text, true);
    push_json_field_f64(&mut s, "entropy_sample", a.entropy_sample, true);
    push_json_array_str(&mut s, "signatures", &a.signatures, true);

    push_json_array_objects(
        &mut s,
        "detected_specs",
        &a.detected_specs,
        true,
        |out, item| {
            push_json_field_str(out, "name", &item.name, true);
            match &item.version {
                Some(v) => push_json_field_str(out, "version", v, true),
                None => push_json_field_null(out, "version", true),
            }
            push_json_field_str(out, "source", &item.source, true);
            push_json_array_str(out, "notes", &item.notes, !item.pattern_matches.is_empty());
            if !item.pattern_matches.is_empty() {
                push_pattern_matches_json(out, &item.pattern_matches, false);
            }
        },
    );

    push_json_array_objects(
        &mut s,
        "detected_models",
        &a.detected_models,
        true,
        |out, item| {
            push_json_field_str(out, "family", &item.family, true);
            match &item.variant {
                Some(v) => push_json_field_str(out, "variant", v, true),
                None => push_json_field_null(out, "variant", true),
            }
            push_json_field_str(
                out,
                "source",
                &item.source,
                !item.pattern_matches.is_empty(),
            );
            if !item.pattern_matches.is_empty() {
                push_pattern_matches_json(out, &item.pattern_matches, false);
            }
        },
    );

    push_json_array_objects(
        &mut s,
        "matched_patterns",
        &a.matched_patterns,
        true,
        |out, item| {
            push_json_field_str(out, "category", &item.category, true);
            push_json_field_str(
                out,
                "source",
                &item.source,
                !item.pattern_matches.is_empty(),
            );
            if !item.pattern_matches.is_empty() {
                push_pattern_matches_json(out, &item.pattern_matches, false);
            }
        },
    );

    push_json_array_objects(
        &mut s,
        "detected_data_structures",
        &a.detected_data_structures,
        true,
        |out, item| {
            push_json_field_str(out, "name", &item.name, true);
            push_json_field_str(out, "source", &item.source, true);
            match item.offset {
                Some(v) => push_json_field_num(out, "offset", v, true),
                None => push_json_field_null(out, "offset", true),
            }
            match item.length {
                Some(v) => push_json_field_num(out, "length", v, true),
                None => push_json_field_null(out, "length", true),
            }
            push_metadata_json_named(
                out,
                "details",
                &item.details,
                !item.pattern_matches.is_empty(),
            );
            if !item.pattern_matches.is_empty() {
                push_pattern_matches_json(out, &item.pattern_matches, false);
            }
        },
    );

    push_json_array_objects(
        &mut s,
        "quantization",
        &a.quantization,
        true,
        |out, item| {
            push_json_field_str(out, "scheme", &item.scheme, true);
            push_json_field_str(
                out,
                "source",
                &item.source,
                !item.pattern_matches.is_empty(),
            );
            if !item.pattern_matches.is_empty() {
                push_pattern_matches_json(out, &item.pattern_matches, false);
            }
        },
    );

    push_json_array_objects(
        &mut s,
        "dataset_size",
        &a.dataset_size,
        true,
        |out, item| {
            push_json_field_str(out, "metric", &item.metric, true);
            push_json_field_str(out, "value", &item.value, true);
            push_json_field_str(
                out,
                "source",
                &item.source,
                !item.pattern_matches.is_empty(),
            );
            if !item.pattern_matches.is_empty() {
                push_pattern_matches_json(out, &item.pattern_matches, false);
            }
        },
    );

    push_json_array_objects(
        &mut s,
        "parameter_data",
        &a.parameter_data,
        true,
        |out, item| {
            push_json_field_str(out, "metric", &item.metric, true);
            push_json_field_str(out, "value", &item.value, true);
            push_json_field_str(
                out,
                "source",
                &item.source,
                !item.pattern_matches.is_empty(),
            );
            if !item.pattern_matches.is_empty() {
                push_pattern_matches_json(out, &item.pattern_matches, false);
            }
        },
    );

    push_json_array_objects(&mut s, "shapes", &a.shapes, true, |out, item| {
        push_json_field_str(out, "kind", &item.kind, true);
        push_json_field_num(out, "offset", item.offset, true);
        match item.length {
            Some(v) => push_json_field_num(out, "length", v, true),
            None => push_json_field_null(out, "length", true),
        }
        push_json_field_str(out, "description", &item.description, true);
        push_json_field_str(out, "source", &item.source, true);
        push_metadata_json_named(
            out,
            "details",
            &item.details,
            !item.pattern_matches.is_empty(),
        );
        if !item.pattern_matches.is_empty() {
            push_pattern_matches_json(out, &item.pattern_matches, false);
        }
    });

    push_metadata_json_named(&mut s, "metadata", &a.metadata, true);
    push_json_array_str(&mut s, "warnings", &a.warnings, false);
    s.push('}');
    maybe_pretty_json(s, pretty)
}

fn push_pattern_matches_json(s: &mut String, values: &[PatternMatch], comma: bool) {
    push_json_array_objects(s, "pattern_matches", values, comma, |out, item| {
        push_json_field_str(out, "pattern", &item.pattern, true);
        push_json_field_num(out, "offset", item.offset, true);
        push_json_field_num(out, "length", item.length, false);
    });
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\u{20}' => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn maybe_pretty_json(compact: String, pretty: bool) -> String {
    if !pretty {
        return compact;
    }
    pretty_print_json(&compact)
}

fn pretty_print_json(compact: &str) -> String {
    let mut out = String::with_capacity(compact.len() + compact.len() / 4);
    let mut indent = 0usize;
    let mut in_string = false;
    let mut escape = false;
    let mut prev_sig: Option<char> = None;

    for ch in compact.chars() {
        if in_string {
            out.push(ch);
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => {
                if matches!(prev_sig, Some('{') | Some('[')) {
                    out.push('\n');
                    push_indent(&mut out, indent);
                }
                in_string = true;
                out.push(ch);
                prev_sig = Some('"');
            }
            '{' | '[' => {
                if matches!(prev_sig, Some('{') | Some('[')) {
                    out.push('\n');
                    push_indent(&mut out, indent);
                }
                out.push(ch);
                indent += 1;
                prev_sig = Some(ch);
            }
            '}' | ']' => {
                indent = indent.saturating_sub(1);
                if !matches!(prev_sig, Some('{') | Some('[')) {
                    out.push('\n');
                    push_indent(&mut out, indent);
                }
                out.push(ch);
                prev_sig = Some(ch);
            }
            ',' => {
                out.push(',');
                out.push('\n');
                push_indent(&mut out, indent);
                prev_sig = Some(',');
            }
            ':' => {
                out.push(':');
                out.push(' ');
                prev_sig = Some(':');
            }
            _ if ch.is_whitespace() => {}
            _ => {
                if matches!(prev_sig, Some('{') | Some('[')) {
                    out.push('\n');
                    push_indent(&mut out, indent);
                }
                out.push(ch);
                prev_sig = Some(ch);
            }
        }
    }

    out
}

fn push_indent(s: &mut String, indent: usize) {
    for _ in 0..indent {
        s.push_str("  ");
    }
}

fn push_json_field_str(s: &mut String, key: &str, value: &str, comma: bool) {
    let _ = write!(s, "\"{}\":\"{}\"", json_escape(key), json_escape(value));
    if comma {
        s.push(',');
    }
}

fn push_json_field_num(s: &mut String, key: &str, value: u64, comma: bool) {
    let _ = write!(s, "\"{}\":{}", json_escape(key), value);
    if comma {
        s.push(',');
    }
}

fn push_json_field_f64(s: &mut String, key: &str, value: f64, comma: bool) {
    let _ = write!(s, "\"{}\":{:.6}", json_escape(key), value);
    if comma {
        s.push(',');
    }
}

fn push_json_field_bool(s: &mut String, key: &str, value: bool, comma: bool) {
    let _ = write!(
        s,
        "\"{}\":{}",
        json_escape(key),
        if value { "true" } else { "false" }
    );
    if comma {
        s.push(',');
    }
}

fn push_json_field_null(s: &mut String, key: &str, comma: bool) {
    let _ = write!(s, "\"{}\":null", json_escape(key));
    if comma {
        s.push(',');
    }
}

fn push_json_array_str(s: &mut String, key: &str, values: &[String], comma: bool) {
    let _ = write!(s, "\"{}\":[", json_escape(key));
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "\"{}\"", json_escape(value));
    }
    s.push(']');
    if comma {
        s.push(',');
    }
}

fn push_json_array_objects<T, F>(s: &mut String, key: &str, values: &[T], comma: bool, mut f: F)
where
    F: FnMut(&mut String, &T),
{
    let _ = write!(s, "\"{}\":[", json_escape(key));
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('{');
        f(s, value);
        s.push('}');
    }
    s.push(']');
    if comma {
        s.push(',');
    }
}

fn push_metadata_json_named(
    s: &mut String,
    key: &str,
    metadata: &BTreeMap<String, String>,
    comma: bool,
) {
    let _ = write!(s, "\"{}\":{{", json_escape(key));
    for (i, (k, v)) in metadata.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "\"{}\":\"{}\"", json_escape(k), json_escape(v));
    }
    s.push('}');
    if comma {
        s.push(',');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_multibyte_scan_does_not_panic_and_reports_pattern_offsets() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&12u64.to_le_bytes());
        bytes.extend_from_slice(&7u64.to_le_bytes());
        bytes.extend_from_slice("llama 7b q4_k ▁token general.architecture".as_bytes());

        let json = analyze_bytes_json("model.gguf", &bytes);
        assert!(json.contains("\"ok\":true"));
        assert!(json.contains("\"GGUF\""));
        assert!(json.contains("\"pattern_matches\""));
        assert!(json.contains("\"offset\":0"));
        assert!(!json.contains("\"confidence\""));
    }

    #[test]
    fn pretty_json_option_adds_newlines() {
        let json =
            analyze_bytes_json_pretty("x.txt", b"{\"model_type\":\"llama\",\"hidden_size\":4096}");
        assert!(json.contains('\n'));
        assert!(json.contains("  \"ok\""));
    }

    #[test]
    fn numeric_suffix_scanner_is_utf8_safe() {
        let matches = find_numeric_suffix_values(
            "42 tokens ▁ 7 b parameters dataset".as_bytes(),
            &[b"tokens", b"b parameters"],
            &[b"dataset", b"parameters"],
            0,
            false,
            8,
        );
        assert!(matches.iter().any(|(value, _)| value == "42 tokens"));
        assert!(matches.iter().any(|(value, _)| value == "7 b parameters"));
    }

    #[test]
    fn false_positive_without_boundary_is_reduced() {
        let json = analyze_bytes_json("blob.bin", b"architecturllama");
        assert!(!json.contains("\"LLaMA\""));
    }

    #[test]
    fn explicit_model_type_value_can_still_identify_t5() {
        let json = analyze_bytes_json(
            "config.json",
            br#"{"model_type":"t5","hidden_size":512,"num_hidden_layers":12}"#,
        );
        assert!(json.contains("\"family\":\"T5\""));
    }

    #[test]
    fn deep_mode_streams_large_safetensors_headers_without_skip_warning() {
        let header =
            br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F16","data_offsets":[0,10]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 10]);

        let mut options = AnalysisOptions::default();
        options.max_safetensors_header_bytes = 16;
        options.deep_scan_chunk_bytes = 32;
        options.deep_scan_overlap_bytes = 16;

        let json = analyze_bytes_json_deep_with_options("model.safetensors", &bytes, &options);
        assert!(json.contains("\"safetensors\""));
        assert!(!json.contains("skipping detailed parse"));
        assert!(json.contains("\"safetensors_root_parse_strategy\":\"streamed_full_header\""));
        assert!(json.contains("\"safetensors_prefix_limit_overridden_in_deep_mode\":\"true\""));
    }

    #[test]
    fn deep_shaper_omits_empty_pattern_match_arrays() {
        let bytes = vec![0u8; 5000];
        let json = analyze_bytes_json_deep("zeros.bin", &bytes);
        assert!(json.contains("\"long_zero_run\""));
        assert!(!json.contains("\"pattern_matches\":[]"));
    }
}
