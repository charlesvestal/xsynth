#![allow(non_camel_case_types)]
use std::{
    collections::{HashMap, HashSet},
    io,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use biquad::Q_BUTTERWORTH_F32;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use thiserror::Error;
use xsynth_soundfonts::{convert_sample_index, FilterType, LoopMode};
use xsynth_soundfonts::sfz::TriggerType;

#[cfg(not(unix))]
use self::audio::load_audio_file;
pub use self::audio::{prebake_audio_cache, AudioLoadError};

use super::{
    voice::VoiceControlData,
    voice::{EnvelopeParameters, Voice},
};
use crate::{helpers::db_to_amp, voice::{CcState, EnvelopeDescriptor}, AudioStreamParams, ChannelCount};

pub use xsynth_soundfonts::{sf2::Sf2ParseError, sfz::SfzParseError};

mod audio;
mod config;
mod sample_storage;
#[cfg(unix)]
mod streaming;
mod utils;
mod voice_spawners;
use utils::*;
use voice_spawners::*;
pub use sample_storage::{MmapHolder, SampleStorage};
#[cfg(unix)]
#[allow(unused_imports)]
pub use streaming::{IoPool, StreamRing, StreamedSampleSource, VoiceStream, HEAD_FRAMES, RING_FRAMES, take_underrun_count, take_underrun_breakdown};

pub use config::*;

pub trait VoiceSpawner: Sync + Send {
    /// Spawn a voice for this region.
    ///
    /// MOVE FORK: `cc_state` is the per-channel raw CC atomic array.
    /// Live `_oncc` voice generators (SIMDVoiceOnccAmp) clone the Arc
    /// to sample CC values per render block. Spawners with no live-CC
    /// bindings ignore it (one Arc::clone refcount increment).
    fn spawn_voice(&self, control: &VoiceControlData, cc_state: &CcState) -> Box<dyn Voice>;
    fn exclusive_class(&self) -> Option<u8> {
        None
    }
}

pub trait SoundfontBase: Sync + Send + std::fmt::Debug {
    fn stream_params(&self) -> &'_ AudioStreamParams;

    fn get_attack_voice_spawners_at(
        &self,
        bank: u8,
        preset: u8,
        key: u8,
        vel: u8,
    ) -> Vec<Box<dyn VoiceSpawner>>;
    fn get_release_voice_spawners_at(
        &self,
        bank: u8,
        preset: u8,
        key: u8,
        vel: u8,
    ) -> Vec<Box<dyn VoiceSpawner>>;
}

#[derive(Clone)]
pub(super) struct LoopParams {
    pub mode: LoopMode,
    pub offset: u32,
    pub start: u32,
    pub end: u32,
    pub stop: Option<u32>,
    /// MOVE FORK / Phase 8: crossfade duration in audio frames.
    /// Zero = no crossfade (legacy behavior — abrupt loop wrap).
    pub crossfade: u32,
}

/// MOVE FORK / 2026-05-16: per-region sample backing. Spawners dispatch
/// on the variant in `begin_voice`:
///   - `Heap`: full sample channels live in RAM (SF2 path + non-unix
///     fallback). Voices read directly from `Arc<SampleStorage>`.
///   - `Streamed`: only the always-resident head buffer is in RAM. The
///     `StreamedSampleSource` carries the open file handle; each voice
///     spawn registers a per-voice ring buffer with the `IoPool`. Used
///     by the SFZ path on unix so libraries larger than RAM (Salamander)
///     are playable.
#[derive(Clone)]
pub(super) enum SampleSource {
    /// Fully-resident sample channels. Voices clone the outer Arc at
    /// spawn time and read i16 samples directly via SampleStorage::get.
    Heap(Arc<[Arc<SampleStorage>]>),
    /// Disk-streamed sample. `source` is shared across all voices that
    /// play this sample; each voice gets its own ring buffer registered
    /// with `io_pool`. `head` is the per-(sample, offset) resident head
    /// buffer (one Arc<[i16]> per channel) that covers frames
    /// [head_start, head_start + head_len) of the file. `head_start`
    /// matches the SFZ region's `offset=` opcode so voices reading at
    /// file-frame `offset` find their data in RAM immediately.
    #[cfg(unix)]
    Streamed {
        source: Arc<streaming::StreamedSampleSource>,
        io_pool: Arc<streaming::IoPool>,
        head: Arc<[Arc<[i16]>]>,
        head_start: u32,
    },
}

impl SampleSource {
    /// Number of audio channels this source carries (1 = mono, 2 = stereo).
    /// Spawners check this against stream_params and fan mono → L/R by
    /// constructing two readers for channel 0 when needed.
    pub fn n_chans(&self) -> usize {
        match self {
            SampleSource::Heap(arr) => arr.len(),
            #[cfg(unix)]
            SampleSource::Streamed { source, .. } => source.n_chans,
        }
    }
}

/// MOVE FORK / 2026-05-16: zero-filled per-channel head fallback. Used
/// when read_head_at fails (e.g. file IO error). Voice gets silence at
/// the start rather than crashing.
#[cfg(unix)]
fn zero_heads(n_chans: usize) -> Arc<[Arc<[i16]>]> {
    let v: Vec<Arc<[i16]>> = (0..n_chans)
        .map(|_| Arc::from(Vec::new().into_boxed_slice()))
        .collect();
    Arc::from(v.into_boxed_slice())
}

struct SampleVoiceSpawnerParams {
    volume: f32,
    pan: f32,
    speed_mult: f32,
    cutoff: Option<f32>,
    resonance: f32,
    /// MOVE FORK: pre-Q base resonance in dB. Needed by live cutoff
    /// modulation to recompute biquad coefficients from
    /// `db_to_amp(base_db + Σ delta·cc/127) * Q_BUTTERWORTH_F32`.
    /// `resonance` (above) is the already-converted Q value used at
    /// filter construction.
    base_resonance_db: f32,
    filter_type: FilterType,
    loop_params: LoopParams,
    envelope: Arc<EnvelopeParameters>,
    sample: SampleSource,
    interpolator: Interpolator,
    exclusive_class: Option<u8>,
    /// MOVE FORK: SFZ round-robin position (1-based) within an RR set.
    /// 0 means "always fire" (no RR). When >0, this spawner only fires
    /// on the NoteOn whose `RrState::counter` lands on this position.
    seq_position: u8,
    /// MOVE FORK: live `volume_oncc<N>=<dB>` bindings from the region.
    /// One Arc shared across all (key, vel) spawners derived from the
    /// same region. Voice gets a cheap Arc::clone at spawn time.
    /// Empty when the region has no volume_oncc opcodes (most SFZ
    /// files; non-DS-converted patches).
    volume_oncc: Arc<[(u8, f32)]>,
    /// MOVE FORK: live `cutoff_oncc<N>=<cents>` bindings from the region.
    /// Same Arc-shared semantics as volume_oncc. Unread by voice spawners
    /// until the SIMD cutoff modulator lands.
    cutoff_oncc: Arc<[(u8, f32)]>,
    /// MOVE FORK: live `resonance_oncc<N>=<dB>` bindings from the region.
    resonance_oncc: Arc<[(u8, f32)]>,
    /// MOVE FORK: live `pan_oncc<N>=<percent>` bindings from the region.
    pan_oncc: Arc<[(u8, f32)]>,
    /// MOVE FORK: Phase 6 curvecc bindings: per-CC (cc, curve_id) for
    /// each `_oncc` family. SIMD generators consult `curves[id][cc_val]`
    /// when a CC has a matching curvecc, else fall back to cc_val/127.
    volume_curvecc: Arc<[(u8, u8)]>,
    cutoff_curvecc: Arc<[(u8, u8)]>,
    resonance_curvecc: Arc<[(u8, u8)]>,
    pan_curvecc: Arc<[(u8, u8)]>,
    /// MOVE FORK: Phase 6 curve tables shared across regions. Empty
    /// HashMap when the SFZ has no `<curve>` blocks.
    curves: Arc<std::collections::HashMap<u8, [f32; 128]>>,
    /// MOVE FORK / Phase 11: amp LFO (sine tremolo). Zero values =
    /// inactive (static_amp short-circuit in SIMDVoiceLfoAmp).
    amp_lfo_freq: f32,
    amp_lfo_depth: f32,
    /// MOVE FORK / Phase 11: live CC mod of amp LFO params.
    amp_lfo_freq_oncc: Arc<[(u8, f32)]>,
    amp_lfo_depth_oncc: Arc<[(u8, f32)]>,
    /// MOVE FORK / Phase 11: filter LFO (cutoff sine, depth cents).
    fil_lfo_freq: f32,
    fil_lfo_depth: f32,
    fil_lfo_freq_oncc: Arc<[(u8, f32)]>,
    fil_lfo_depth_oncc: Arc<[(u8, f32)]>,
    /// MOVE FORK / Phase 11: pan LFO (pan sine, depth percent).
    pan_lfo_freq: f32,
    pan_lfo_depth: f32,
    pan_lfo_freq_oncc: Arc<[(u8, f32)]>,
    pan_lfo_depth_oncc: Arc<[(u8, f32)]>,
    /// MOVE FORK / Phase 11: filter envelope (autowah). depth cents at peak.
    fileg_attack: f32,
    fileg_decay: f32,
    fileg_sustain: f32,
    fileg_release: f32,
    fileg_depth: f32,
    /// MOVE FORK / 2026-05-16: optional `<curve>` table index for the
    /// filter envelope. When Some, LiveCutoffState uses
    /// `curves[id][env_level*127]` as a 0..1 multiplier on `fileg_depth`,
    /// so the converter can pre-bake DS's linear-Hz envelope sweep into
    /// the exp-cents domain.
    fileg_curve: Option<u8>,
    /// MOVE FORK / Phase 11: pitch LFO (vibrato). depth in cents.
    pitch_lfo_freq: f32,
    pitch_lfo_depth: f32,
    pitch_lfo_freq_oncc: Arc<[(u8, f32)]>,
    pitch_lfo_depth_oncc: Arc<[(u8, f32)]>,
}

pub(super) struct SoundfontInstrument {
    bank: u8,
    preset: u8,
    spawner_params_list: Vec<Vec<Arc<SampleVoiceSpawnerParams>>>,
    /// MOVE FORK: release-trigger spawners, segregated from attack so
    /// `trigger=release` SFZ regions fire on NoteOff and stay silent on
    /// NoteOn. xsynth's get_release_voice_spawners_at was previously a
    /// stub returning empty — now it returns these.
    release_spawner_params_list: Vec<Vec<Arc<SampleVoiceSpawnerParams>>>,
    /// MOVE FORK: round-robin counters, one per (key, vel) slot. None
    /// means "no RR at this slot" (the attack-spawner list fires
    /// unfiltered). Some({ seq_length, counter }) means each NoteOn
    /// increments counter and only spawners whose `seq_position`
    /// matches (counter % seq_length + 1) fire. seq_position == 0
    /// always fires regardless.
    rr_state: Vec<Option<RrState>>,
}

pub(super) struct RrState {
    seq_length: u8,
    counter: std::sync::atomic::AtomicU32,
}

/// Represents a sample soundfont to be used within XSynth.
///
/// Supports SFZ and SF2 soundfonts.
///
/// ## SFZ specification support (opcodes)
/// - `lovel` & `hivel`
/// - `lokey` & `hikey`
/// - `pitch_keycenter`
/// - `volume`
/// - `pan`
/// - `sample`
/// - `default_path`
/// - `loop_mode`
/// - `loop_start`
/// - `loop_end`
/// - `offset`
/// - `cutoff`
/// - `resonance`
/// - `fil_veltrack`
/// - `fil_keycenter`
/// - `fil_keytrack`
/// - `filter_type`
/// - `tune`
/// - `ampeg_start`
/// - `ampeg_delay`
/// - `ampeg_attack`
/// - `ampeg_hold`
/// - `ampeg_decay`
/// - `ampeg_sustain`
/// - `ampeg_release`
///
/// ## SF2 specification support
/// ### Generators
/// - `startAddrsOffset`
/// - `endAddrsOffset`
/// - `startloopAddrsOffset`
/// - `endloopAddrsOffset`
/// - `startAddrsCoarseOffset`
/// - `endAddrsCoarseOffset`
/// - `startloopAddrsCoarseOffset`
/// - `endloopAddrsCoarseOffset`
/// - `initialFilterFc`
/// - `initialFilterQ`
/// - `pan`
/// - `delayVolEnv`
/// - `attackVolEnv`
/// - `holdVolEnv`
/// - `decayVolEnv`
/// - `sustainVolEnv`
/// - `releaseVolEnv`
/// - `instrument`
/// - `keyRange`
/// - `velRange`
/// - `initialAttenuation`
/// - `coarseTune`
/// - `fineTune`
/// - `sampleID`
/// - `sampleModes`
/// - `overridingRootKey`
/// - `scaleTuning`
/// - `exclusiveClass`
/// - `keynum`
/// - `velocity`
/// - `keynumToVolEnvHold`
/// - `keynumToVolEnvDecay`
///
/// ### Modulators
/// XSynth intentionally supports a baked subset of SF2 modulators so note-on
/// articulation can be resolved at soundfont load time without adding runtime
/// voice stages. Currently supported:
/// - sources: key number and note-on velocity
/// - curves: linear, concave, convex, and switch
/// - transforms: linear and absolute
/// - destinations: attenuation, filter cutoff, pan, volume envelope timings,
///   and static pitch offsets
///
/// The following SF2 features are intentionally not supported in the runtime
/// engine because they would add significant hot-path and binary-size cost:
/// - modulation envelope generators and destinations
/// - modulation LFO / vibrato LFO generators and destinations
/// - generic CC / aftertouch / pitch-wheel-driven SF2 modulators
/// - chorus and reverb send behavior
pub struct SampleSoundfont {
    instruments: Vec<SoundfontInstrument>,
    stream_params: AudioStreamParams,
    /// MOVE FORK / 2026-05-17: max absolute f32 amplitude across all
    /// per-region head buffers, factoring in each region's `volume`
    /// multiplier. Used by the C plugin to compute a per-preset
    /// attenuation that brings hot SFZ libraries (Salamander piano:
    /// per-voice peak ≈ 0.82) into the same loudness ballpark as
    /// DecentSampler without forcing the user to manage per-preset
    /// gain knobs.
    estimated_voice_peak: f32,
}

/// Errors that can be generated when loading an SFZ soundfont.
#[derive(Debug, Error)]
pub enum LoadSfzError {
    #[error("IO Error")]
    IOError(#[from] io::Error),

    #[error("Error loading samples")]
    AudioLoadError(#[from] AudioLoadError),

    #[error("Error parsing the SFZ: {0}")]
    SfzParseError(#[from] SfzParseError),

    /// MOVE FORK: cancellation signal observed.
    #[error("Load was cancelled")]
    Cancelled,
}

/// Errors that can be generated when loading a soundfont
/// of an unspecified format.
#[derive(Debug, Error)]
pub enum LoadSfError {
    #[error("Error loading the SFZ: {0}")]
    LoadSfzError(#[from] LoadSfzError),

    #[error("Error loading the SF2: {0}")]
    LoadSf2Error(#[from] Sf2ParseError),

    #[error("Unsupported format")]
    Unsupported,
}

impl SampleSoundfont {
    /// Loads a new sample soundfont of an unspecified type.
    /// The type of the soundfont will be determined from the file extension.
    ///
    /// Parameters:
    /// - `path`: The path of the soundfont to be loaded.
    /// - `stream_params`: Parameters of the output audio. See the `AudioStreamParams`
    ///   documentation for the available options.
    /// - `options`: The soundfont configuration. See the `SoundfontInitOptions`
    ///   documentation for the available options.
    pub fn new(
        path: impl Into<PathBuf>,
        stream_params: AudioStreamParams,
        options: SoundfontInitOptions,
    ) -> Result<Self, LoadSfError> {
        let path: PathBuf = path.into();
        if let Some(ext) = path.extension() {
            match ext.to_str().unwrap_or("").to_lowercase().as_str() {
                "sfz" => {
                    Self::new_sfz(path, stream_params, options).map_err(LoadSfError::LoadSfzError)
                }
                "sf2" => {
                    Self::new_sf2(path, stream_params, options).map_err(LoadSfError::LoadSf2Error)
                }
                _ => Err(LoadSfError::Unsupported),
            }
        } else {
            Err(LoadSfError::Unsupported)
        }
    }

    /// MOVE FORK: cancellable variant. Pass an AtomicBool that the caller can
    /// flip to abort the load between sample decodes.
    pub fn new_sfz_cancellable(
        sfz_path: impl Into<PathBuf>,
        stream_params: AudioStreamParams,
        options: SoundfontInitOptions,
        cancel: Option<Arc<AtomicBool>>,
    ) -> Result<Self, LoadSfzError> {
        Self::new_sfz_inner(sfz_path.into(), stream_params, options, cancel)
    }

    /// Loads a new SFZ soundfont
    pub fn new_sfz(
        sfz_path: impl Into<PathBuf>,
        stream_params: AudioStreamParams,
        options: SoundfontInitOptions,
    ) -> Result<Self, LoadSfzError> {
        Self::new_sfz_inner(sfz_path.into(), stream_params, options, None)
    }

    fn new_sfz_inner(
        sfz_path: PathBuf,
        stream_params: AudioStreamParams,
        options: SoundfontInitOptions,
        cancel: Option<Arc<AtomicBool>>,
    ) -> Result<Self, LoadSfzError> {
        let check_cancel = || -> Result<(), LoadSfzError> {
            if let Some(c) = &cancel {
                if c.load(Ordering::Relaxed) { return Err(LoadSfzError::Cancelled); }
            }
            Ok(())
        };
        check_cancel()?;
        let regions = xsynth_soundfonts::sfz::parse_soundfont(sfz_path)?;

        // MOVE FORK / 2026-05-16: SFZ keyswitch v1 filter. The SFZ spec's
        // `sw_default` sets the initial keyswitch key for the file;
        // `sw_last` on a region constrains it to fire only when the
        // most-recent keyswitch press matches that key. We don't yet
        // track live keyswitch state per channel, so for now: pick the
        // first sw_default seen, drop any region whose sw_last is set
        // and doesn't match it. Lets Salamander Grand Piano load with
        // only the Natural master active (the Retuned master's regions
        // are tagged sw_last=$RETUNED and get filtered out).
        let effective_keyswitch: Option<i8> = regions
            .iter()
            .find_map(|r| r.sw_default);
        let regions: Vec<_> = if let Some(ks) = effective_keyswitch {
            regions
                .into_iter()
                .filter(|r| match r.sw_last {
                    Some(last) => last == ks,
                    None => true,
                })
                .collect()
        } else {
            regions
        };

        // Find the unique samples that we need to parse and convert
        let unique_sample_params: HashSet<_> = regions
            .iter()
            .map(sample_cache_from_region_params)
            .collect();

        // MOVE FORK: sample loading is sequential on Move. The original
        // `into_par_iter()` decodes every sample via rayon at once, which on
        // 452-sample patches (WörliTzer) ballooned peak memory and aborted
        // the process via Rust's alloc handler on our ~1.5 GB device. Serial
        // load keeps peak well under budget; render-time rayon parallelism
        // is untouched.
        //
        // MOVE FORK / 2026-05-16: on unix we always go disk-streamed —
        // memory cost is bounded to the head buffer (~1 s per sample
        // per unique region.offset) plus active voice rings. Heap
        // residency is structurally unsupportable for libraries like
        // Salamander Grand Piano (~1.1 GB decoded i16). The IoPool is
        // per-soundfont; it lives in the SampleSource::Streamed Arc
        // carried by every spawner, so it goes away when the last
        // spawner drops.
        #[cfg(unix)]
        let io_pool: Arc<streaming::IoPool> = streaming::IoPool::new();

        // Phase A: load each unique sample's metadata (file handle +
        // layout). On unix this is the streamed source; on other
        // platforms it's a heap-loaded Arc<[Arc<SampleStorage>]>.
        let total_samples_count = unique_sample_params.len();
        let progress_start = std::time::Instant::now();
        let mut samples_loaded: usize = 0;

        #[cfg(unix)]
        let mut sources: HashMap<_, (Arc<streaming::StreamedSampleSource>, u32)> =
            HashMap::with_capacity(total_samples_count);
        #[cfg(not(unix))]
        let mut heap_samples: HashMap<_, (Arc<[Arc<SampleStorage>]>, u32)> =
            HashMap::with_capacity(total_samples_count);

        for params in unique_sample_params.iter() {
            check_cancel()?;
            #[cfg(unix)]
            {
                let (source, rate) =
                    audio::load_audio_file_streamed(&params.path, stream_params)?;
                sources.insert(params.clone(), (source, rate));
            }
            #[cfg(not(unix))]
            {
                let (channels, rate) = load_audio_file(&params.path, stream_params)?;
                heap_samples.insert(params.clone(), (channels, rate));
            }
            samples_loaded += 1;
            if samples_loaded == 1
                || samples_loaded == total_samples_count
                || samples_loaded % 16 == 0
            {
                let elapsed = progress_start.elapsed().as_secs_f64();
                let eta = if samples_loaded > 0 {
                    elapsed * (total_samples_count - samples_loaded) as f64
                        / samples_loaded as f64
                } else {
                    0.0
                };
                let line = format!(
                    "[xsynth] sample load: {}/{} elapsed={:.1}s eta={:.1}s last={}\n",
                    samples_loaded,
                    total_samples_count,
                    elapsed,
                    eta,
                    params.path.display(),
                );
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("/data/UserData/schwung/tmp/xsynth_debug.log")
                {
                    let _ = f.write_all(line.as_bytes());
                }
            }
        }

        // Phase B (unix only): pre-load per-(sample, region.offset)
        // head buffers. SFZ regions can carry an `offset=N` opcode that
        // skips N frames of the source file. Voices reading at file
        // position N need the head buffer to start at N — not at frame
        // 0 — so the audio thread doesn't hit an empty ring on noteon.
        // Heads dedupe across regions sharing the same (path, offset)
        // pair. RAM per head: HEAD_FRAMES × n_chans × 2 bytes.
        #[cfg(unix)]
        let head_cache: HashMap<_, Arc<[Arc<[i16]>]>> = {
            // MOVE FORK / 2026-05-17: per-(sample,offset) max head length.
            // For looped regions, the head must cover [target_offset,
            // loop_end] so that loop-wrap reads (which land below the
            // ring's trailing RING_FRAMES window when the loop region
            // exceeds RING_FRAMES) always hit the always-resident head
            // and never the bounded ring.
            let mut head_lens: HashMap<_, usize> = HashMap::new();
            for region in &regions {
                let key = sample_cache_from_region_params(region);
                // MOVE FORK / 2026-05-17 (day 3): offset is expressed in
                // source-rate frames (SFZ spec). Convert to the ring's
                // data rate — for cache layout that's target_rate, for
                // WAV/FLAC streaming that's the source's native rate.
                let (src_rate, data_rate) = sources.get(&key)
                    .map(|(s, r)| (*r, s.data_rate))
                    .unwrap_or((stream_params.sample_rate, stream_params.sample_rate));
                let target_offset = convert_sample_index(
                    region.offset,
                    src_rate,
                    data_rate,
                );
                // Region's required head length: HEAD_FRAMES for plain
                // play-through, OR loop_end - target_offset for looped
                // regions so the loop region itself is fully resident.
                let mut needed = streaming::HEAD_FRAMES;
                let region_loops = region.loop_start != region.loop_end
                    && matches!(region.loop_mode,
                        LoopMode::LoopContinuous | LoopMode::LoopSustain);
                if region_loops {
                    let loop_end_target = convert_sample_index(
                        region.loop_end,
                        src_rate,
                        stream_params.sample_rate,
                    ) as usize;
                    if loop_end_target > target_offset as usize {
                        let span = loop_end_target - target_offset as usize + 1;
                        if span > needed { needed = span; }
                    }
                }
                let entry = head_lens.entry((key, target_offset)).or_insert(0);
                if needed > *entry { *entry = needed; }
            }
            let mut map = HashMap::with_capacity(head_lens.len());
            for ((key, target_offset), head_len) in head_lens {
                check_cancel()?;
                if let Some(src) = sources.get(&key).map(|(s, _)| s.clone()) {
                    let head = src
                        .read_head_at_len(target_offset as usize, head_len)
                        .unwrap_or_else(|| {
                            (0..src.n_chans)
                                .map(|_| Arc::from(Vec::new().into_boxed_slice()))
                                .collect()
                        });
                    let head_arc: Arc<[Arc<[i16]>]> = Arc::from(head.into_boxed_slice());
                    map.insert((key, target_offset), head_arc);
                }
            }
            map
        };

        // Generate region params. MOVE FORK: parallel attack + release lists.
        let mut spawner_params_list = Vec::<Vec<Arc<SampleVoiceSpawnerParams>>>::new();
        let mut release_spawner_params_list = Vec::<Vec<Arc<SampleVoiceSpawnerParams>>>::new();
        // MOVE FORK: track max seq_length per (key, vel) so we can build
        // rr_state with the right rotation period after the loop.
        let mut rr_seq_length = vec![0u8; 128 * 128];
        for _ in 0..(128 * 128) {
            spawner_params_list.push(Vec::new());
            release_spawner_params_list.push(Vec::new());
        }

        // Write region params
        for region in regions {
            let params = sample_cache_from_region_params(&region);

            // Key value -1 is used for CC triggered regions which are not supported by XSynth
            if region.keyrange.contains(&-1) {
                continue;
            }

            // MOVE FORK / 2026-05-17 (day 3): if this region's sample is
            // backed by a streamed source whose data is at a different
            // rate than the target, fold that rate ratio into speed_mult
            // so the voice's linear interpolator resamples on the fly.
            // Cache (Separated) layouts always have data_rate ==
            // target_rate so the ratio is 1.0 — no behavior change.
            let data_rate_ratio: f32 = {
                #[cfg(unix)]
                {
                    sources.get(&params).map(|(s, _)| {
                        s.data_rate as f32 / stream_params.sample_rate as f32
                    }).unwrap_or(1.0)
                }
                #[cfg(not(unix))]
                {
                    1.0
                }
            };
            // Diagnostic: log when ratio != 1.0 (i.e. SRC kicks in).
            if (data_rate_ratio - 1.0).abs() > 1e-4 {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true).append(true)
                    .open("/data/UserData/schwung/tmp/xsynth_debug.log")
                {
                    let _ = writeln!(f,
                        "[xsynth] SRC region: path={} ratio={:.4}",
                        params.path.display(), data_rate_ratio);
                }
            }

            // MOVE FORK: one Arc per region for live `_oncc` bindings,
            // cloned cheaply into every (key, vel) spawner.
            let volume_oncc: Arc<[(u8, f32)]> = region.volume_oncc.clone().into();
            let cutoff_oncc: Arc<[(u8, f32)]> = region.cutoff_oncc.clone().into();
            let resonance_oncc: Arc<[(u8, f32)]> = region.resonance_oncc.clone().into();
            let pan_oncc: Arc<[(u8, f32)]> = region.pan_oncc.clone().into();
            let volume_curvecc: Arc<[(u8, u8)]> = region.volume_curvecc.clone().into();
            let cutoff_curvecc: Arc<[(u8, u8)]> = region.cutoff_curvecc.clone().into();
            let resonance_curvecc: Arc<[(u8, u8)]> = region.resonance_curvecc.clone().into();
            let pan_curvecc: Arc<[(u8, u8)]> = region.pan_curvecc.clone().into();
            let curves = region.curves.clone();

            for key in region.keyrange.clone() {
                for vel in region.velrange.clone() {
                    let index = key_vel_to_index(key as u8, vel);
                    let speed_mult =
                        get_speed_mult_from_keys(key as u8, region.pitch_keycenter as u8)
                            * cents_factor(region.tune as f32)
                            * data_rate_ratio;

                    let mut envelope = region.ampeg_envelope.clone();
                    envelope.ampeg_release +=
                        (vel as f32 / 127.0) * region.ampeg_envelope.ampeg_vel2release;
                    let envelope_params = Arc::new(
                        envelope_descriptor_from_region_params(&envelope).to_envelope_params(
                            stream_params.sample_rate,
                            options.vol_envelope_options,
                        ),
                    );

                    let mut cutoff = None;
                    if options.use_effects {
                        if let Some(mut cutoff_t) = region.cutoff {
                            if cutoff_t >= 1.0 {
                                let cents = vel as f32 / 127.0 * region.fil_veltrack as f32
                                    + (key as f32 - region.fil_keycenter as f32)
                                        * region.fil_keytrack as f32;
                                cutoff_t *= cents_factor(cents);
                                cutoff = Some(
                                    cutoff_t
                                        .clamp(1.0, stream_params.sample_rate as f32 / 2.0 - 100.0),
                                );
                            }
                        }
                    }

                    let pan_mult = vel as f32 / 127.0 * region.pan_veltrack
                        + (key as f32 - region.pan_keycenter as f32) * region.pan_keytrack;
                    let pan = (region.pan as f32 + pan_mult).clamp(-100.0, 100.0) / 100.0;
                    let pan = (pan + 1.0) / 2.0;

                    let vol_vel = {
                        let a = region.amp_veltrack / 100.0;
                        let aabs = a.abs();
                        let vel = vel as f32;

                        127.0 * (1.0 - aabs)
                            + vel * (a + aabs) / 2.0
                            + (127.0 - vel) * (aabs - a) / 2.0
                    };
                    // MOVE FORK / 2026-05-16: DS uses a purely linear
                    // velocity → amp curve. Fine-grid probe (vel 1, 5,
                    // 10, 16, 24, 32, 40, 50, 60, 64, 70, 80, 90, 100,
                    // 110, 120, 127) showed amp = vel/127 to four
                    // decimal places at every step. xsynth's SF2-style
                    // square was wrong for our DS use case; linear
                    // produces exact DS amp parity when paired with
                    // amp_veltrack=100 in the converter.
                    let vol_mult = vol_vel / 127.0;
                    let vol_db_add =
                        (key as f32 - region.amp_keycenter as f32) * region.amp_keytrack;
                    let vol_db = (region.volume as f32 + vol_db_add).clamp(-96.0, 12.0);
                    let volume = vol_mult * db_to_amp(vol_db);

                    #[cfg(unix)]
                    let sample_rate = sources[&params].1;
                    #[cfg(not(unix))]
                    let sample_rate = heap_samples[&params].1;

                    let loop_params = LoopParams {
                        mode: if region.loop_start == region.loop_end {
                            LoopMode::NoLoop
                        } else {
                            region.loop_mode
                        },
                        offset: convert_sample_index(
                            region.offset,
                            sample_rate,
                            stream_params.sample_rate,
                        ),
                        start: convert_sample_index(
                            region.loop_start,
                            sample_rate,
                            stream_params.sample_rate,
                        ),
                        end: convert_sample_index(
                            region.loop_end,
                            sample_rate,
                            stream_params.sample_rate,
                        ),
                        stop: None,
                        // MOVE FORK / Phase 8: SFZ `loop_crossfade` is
                        // in seconds — multiply by output rate, not the
                        // source rate, since the sampler indexes into
                        // the resampled stream.
                        crossfade: (region.loop_crossfade
                            * stream_params.sample_rate as f32) as u32,
                    };

                    // MOVE FORK / 2026-05-16: per-region sample backing.
                    // Heap: clone the Arc<[Arc<SampleStorage>]>; fan mono
                    // to stereo via Arc duplication. Streamed: clone the
                    // (source, io_pool, per-(path,offset) head, head_start).
                    // The mono→stereo fan-out happens at voice-spawn time.
                    #[cfg(unix)]
                    let region_samples: SampleSource = {
                        let (src, _) = &sources[&params];
                        let target_offset = loop_params.offset;
                        let head = head_cache
                            .get(&(params.clone(), target_offset))
                            .cloned()
                            .unwrap_or_else(|| zero_heads(src.n_chans));
                        SampleSource::Streamed {
                            source: src.clone(),
                            io_pool: io_pool.clone(),
                            head,
                            head_start: target_offset,
                        }
                    };
                    #[cfg(not(unix))]
                    let region_samples: SampleSource = {
                        let mut a = heap_samples[&params].0.clone();
                        if stream_params.channels == ChannelCount::Stereo && a.len() == 1 {
                            a = Arc::new([a[0].clone(), a[0].clone()]);
                        }
                        SampleSource::Heap(a)
                    };

                    let spawner_params = Arc::new(SampleVoiceSpawnerParams {
                        pan,
                        volume,
                        envelope: envelope_params,
                        speed_mult,
                        cutoff,
                        resonance: db_to_amp(region.resonance) * Q_BUTTERWORTH_F32,
                        base_resonance_db: region.resonance,
                        filter_type: region.filter_type,
                        interpolator: options.interpolator,
                        loop_params,
                        sample: region_samples,
                        exclusive_class: None,
                        seq_position: region.seq_position.min(u8::MAX as u32) as u8,
                        volume_oncc: volume_oncc.clone(),
                        cutoff_oncc: cutoff_oncc.clone(),
                        resonance_oncc: resonance_oncc.clone(),
                        pan_oncc: pan_oncc.clone(),
                        volume_curvecc: volume_curvecc.clone(),
                        cutoff_curvecc: cutoff_curvecc.clone(),
                        resonance_curvecc: resonance_curvecc.clone(),
                        pan_curvecc: pan_curvecc.clone(),
                        curves: curves.clone(),
                        amp_lfo_freq: region.amp_lfo_freq,
                        amp_lfo_depth: region.amp_lfo_depth,
                        amp_lfo_freq_oncc: region.amp_lfo_freq_oncc.clone().into(),
                        amp_lfo_depth_oncc: region.amp_lfo_depth_oncc.clone().into(),
                        fil_lfo_freq: region.fil_lfo_freq,
                        fil_lfo_depth: region.fil_lfo_depth,
                        fil_lfo_freq_oncc: region.fil_lfo_freq_oncc.clone().into(),
                        fil_lfo_depth_oncc: region.fil_lfo_depth_oncc.clone().into(),
                        pan_lfo_freq: region.pan_lfo_freq,
                        pan_lfo_depth: region.pan_lfo_depth,
                        pan_lfo_freq_oncc: region.pan_lfo_freq_oncc.clone().into(),
                        pan_lfo_depth_oncc: region.pan_lfo_depth_oncc.clone().into(),
                        fileg_attack:  region.fileg_attack,
                        fileg_decay:   region.fileg_decay,
                        fileg_sustain: region.fileg_sustain,
                        fileg_release: region.fileg_release,
                        fileg_depth:   region.fileg_depth,
                        fileg_curve:   region.fileg_curve,
                        pitch_lfo_freq:  region.pitch_lfo_freq,
                        pitch_lfo_depth: region.pitch_lfo_depth,
                        pitch_lfo_freq_oncc:  region.pitch_lfo_freq_oncc.clone().into(),
                        pitch_lfo_depth_oncc: region.pitch_lfo_depth_oncc.clone().into(),
                    });

                    // MOVE FORK: track max seq_length seen at this slot
                    // so rr_state has the right rotation period. Regions
                    // that share seq_length at one (key, vel) form one
                    // RR set; any non-zero seq_length triggers RR.
                    if region.seq_length > 0 {
                        let v = region.seq_length.min(u8::MAX as u32) as u8;
                        if rr_seq_length[index] < v {
                            rr_seq_length[index] = v;
                        }
                    }

                    match region.trigger {
                        TriggerType::Release => release_spawner_params_list[index]
                            .push(spawner_params.clone()),
                        TriggerType::Attack => spawner_params_list[index]
                            .push(spawner_params.clone()),
                    }
                }
            }
        }

        // MOVE FORK: materialize per-slot RR state from the tracked
        // seq_lengths. None for slots without any RR region.
        let rr_state: Vec<Option<RrState>> = rr_seq_length
            .into_iter()
            .map(|len| {
                if len == 0 {
                    None
                } else {
                    Some(RrState {
                        seq_length: len,
                        counter: std::sync::atomic::AtomicU32::new(0),
                    })
                }
            })
            .collect();

        // MOVE FORK / 2026-05-17: per-preset peak estimate for auto-gain.
        // For each spawner_params (one per region × key × vel slot),
        // estimate worst-case voice peak = sample_peak × params.volume.
        // The plugin uses this to compute a per-preset attenuation
        // that brings hot SFZ libs into the same loudness ballpark as
        // DecentSampler.
        //
        // For streamed sources, sample peak is read from the resident
        // head buffer (first ~1 s of each sample, which covers piano
        // attacks). For heap sources, scan the full Arc<[i16]> data.
        #[cfg(unix)]
        let estimated_voice_peak: f32 = {
            let mut max_peak: f32 = 0.0;
            for slot in spawner_params_list.iter().chain(release_spawner_params_list.iter()) {
                for sp in slot.iter() {
                    let sample_peak_i16: i16 = match &sp.sample {
                        SampleSource::Streamed { head, .. } => head
                            .iter()
                            .flat_map(|ch| ch.iter().copied())
                            .map(|v| v.unsigned_abs() as i16)
                            .max()
                            .unwrap_or(0),
                        SampleSource::Heap(arr) => arr
                            .iter()
                            .flat_map(|ch| (0..ch.len()).map(|i| ch.get(i).unsigned_abs() as i16))
                            .max()
                            .unwrap_or(0),
                    };
                    let sample_peak_f32 = sample_peak_i16 as f32 / 32768.0;
                    let voice_peak = sample_peak_f32 * sp.volume.abs();
                    if voice_peak > max_peak {
                        max_peak = voice_peak;
                    }
                }
            }
            max_peak
        };
        #[cfg(not(unix))]
        let estimated_voice_peak: f32 = 1.0; // Be conservative on non-unix builds.

        Ok(SampleSoundfont {
            instruments: vec![SoundfontInstrument {
                bank: options.bank.unwrap_or(0),
                preset: options.preset.unwrap_or(0),
                spawner_params_list,
                release_spawner_params_list,
                rr_state,
            }],
            stream_params,
            estimated_voice_peak,
        })
    }

    /// Loads a new SF2 soundfont
    ///
    /// Parameters:
    /// - `path`: The path of the SF2 soundfont to be loaded.
    /// - `stream_params`: Parameters of the output audio. See the `AudioStreamParams`
    ///   documentation for the available options.
    /// - `options`: The soundfont configuration. See the `SoundfontInitOptions`
    ///   documentation for the available options.
    pub fn new_sf2(
        sf2_path: impl Into<PathBuf>,
        stream_params: AudioStreamParams,
        options: SoundfontInitOptions,
    ) -> Result<Self, Sf2ParseError> {
        let presets =
            xsynth_soundfonts::sf2::load_soundfont(sf2_path.into(), stream_params.sample_rate)?;

        let mut instruments = Vec::new();

        for preset in presets {
            if let Some(bank) = options.bank {
                if bank != preset.bank as u8 {
                    continue;
                }
            }
            if let Some(presetn) = options.preset {
                if presetn != preset.preset as u8 {
                    continue;
                }
            }

            // MOVE FORK: SF2 has no release-trigger concept; the release
            // list is always empty here. Kept parallel to SFZ path for
            // SoundfontInstrument's uniform shape.
            let mut spawner_params_list = Vec::<Vec<Arc<SampleVoiceSpawnerParams>>>::new();
            let release_spawner_params_list = vec![Vec::new(); 128 * 128];
            for _ in 0..(128 * 128) {
                spawner_params_list.push(Vec::new());
            }

            let mut unique_envelope_params =
                Vec::<(EnvelopeDescriptor, Arc<EnvelopeParameters>)>::new();

            for region in preset.regions {
                for key in region.keyrange.clone() {
                    for vel in region.velrange.clone() {
                        let index = key_vel_to_index(key, vel);
                        let note_params = region.note_params(key, vel);
                        let envelope =
                            envelope_descriptor_from_region_params(&note_params.ampeg_envelope);
                        let envelope_params = if let Some((_, params)) = unique_envelope_params
                            .iter()
                            .find(|(descriptor, _)| *descriptor == envelope)
                        {
                            params.clone()
                        } else {
                            let params = Arc::new(envelope.to_envelope_params(
                                stream_params.sample_rate,
                                options.vol_envelope_options,
                            ));
                            unique_envelope_params.push((envelope, params.clone()));
                            params
                        };
                        let tuned_key_cents =
                            (key as f32 - region.root_key as f32) * region.scale_tuning as f32;
                        let speed_mult = cents_factor(
                            tuned_key_cents
                                + region.fine_tune as f32
                                + region.coarse_tune as f32 * 100.0
                                + note_params.tune_cents,
                        );

                        let mut cutoff = None;
                        if options.use_effects {
                            if let Some(cutoff_t) = note_params.cutoff {
                                if cutoff_t >= 1.0 {
                                    cutoff = Some(cutoff_t.clamp(
                                        1.0,
                                        stream_params.sample_rate as f32 / 2.0 - 100.0,
                                    ));
                                }
                            }
                        }

                        let pan = ((note_params.pan as f32 / 500.0) + 1.0) / 2.0;

                        let loop_params = LoopParams {
                            mode: if region.loop_start == region.loop_end {
                                LoopMode::NoLoop
                            } else {
                                region.loop_mode
                            },
                            offset: region.offset,
                            start: region.loop_start,
                            end: region.loop_end,
                            stop: Some(region.sample_end),
                            crossfade: 0, // SF2 has no loop_crossfade
                        };

                        let mut region_samples: Arc<[Arc<[i16]>]> = region.sample.clone();
                        if stream_params.channels == ChannelCount::Stereo
                            && region_samples.len() == 1
                        {
                            region_samples =
                                Arc::new([region_samples[0].clone(), region_samples[0].clone()]);
                        }
                        // MOVE FORK: SF2 ships Arc<[Arc<[i16]>]>; wrap each
                        // per-channel buffer in SampleStorage so it matches
                        // the unified SampleVoiceSpawnerParams type.
                        let sample_storage: Arc<[Arc<SampleStorage>]> = region_samples
                            .iter()
                            .map(|c| Arc::new(SampleStorage::from_heap(c.clone())))
                            .collect();

                        let spawner_params = Arc::new(SampleVoiceSpawnerParams {
                            pan,
                            volume: note_params.volume,
                            envelope: envelope_params,
                            speed_mult,
                            cutoff,
                            resonance: db_to_amp(note_params.resonance) * Q_BUTTERWORTH_F32,
                            base_resonance_db: note_params.resonance,
                            filter_type: FilterType::LowPass,
                            interpolator: options.interpolator,
                            loop_params,
                            // MOVE FORK / 2026-05-16: SF2 stays heap-resident.
                            // Its sample data is unpacked from one big binary
                            // blob at load time and doesn't have the per-sample
                            // file structure that the streaming path requires.
                            // SF2 isn't used by schwung-sfz anyway; this is
                            // upstream-compat plumbing.
                            sample: SampleSource::Heap(sample_storage),
                            exclusive_class: region.exclusive_class,
                            seq_position: 0, // SF2 has no round-robin concept
                            // SF2 has no `_oncc` surface; empty Arcs.
                            volume_oncc: Arc::from(Vec::<(u8, f32)>::new()),
                            cutoff_oncc: Arc::from(Vec::<(u8, f32)>::new()),
                            resonance_oncc: Arc::from(Vec::<(u8, f32)>::new()),
                            pan_oncc: Arc::from(Vec::<(u8, f32)>::new()),
                            volume_curvecc: Arc::from(Vec::<(u8, u8)>::new()),
                            cutoff_curvecc: Arc::from(Vec::<(u8, u8)>::new()),
                            resonance_curvecc: Arc::from(Vec::<(u8, u8)>::new()),
                            pan_curvecc: Arc::from(Vec::<(u8, u8)>::new()),
                            curves: Arc::new(std::collections::HashMap::new()),
                            amp_lfo_freq: 0.0,
                            amp_lfo_depth: 0.0,
                            amp_lfo_freq_oncc: Arc::from(Vec::<(u8, f32)>::new()),
                            amp_lfo_depth_oncc: Arc::from(Vec::<(u8, f32)>::new()),
                            fil_lfo_freq: 0.0,
                            fil_lfo_depth: 0.0,
                            fil_lfo_freq_oncc: Arc::from(Vec::<(u8, f32)>::new()),
                            fil_lfo_depth_oncc: Arc::from(Vec::<(u8, f32)>::new()),
                            pan_lfo_freq: 0.0,
                            pan_lfo_depth: 0.0,
                            pan_lfo_freq_oncc: Arc::from(Vec::<(u8, f32)>::new()),
                            pan_lfo_depth_oncc: Arc::from(Vec::<(u8, f32)>::new()),
                            fileg_attack: 0.0,
                            fileg_decay: 0.0,
                            fileg_sustain: 1.0,
                            fileg_release: 0.0,
                            fileg_depth: 0.0,
                            fileg_curve: None,
                            pitch_lfo_freq: 0.0,
                            pitch_lfo_depth: 0.0,
                            pitch_lfo_freq_oncc: Arc::from(Vec::<(u8, f32)>::new()),
                            pitch_lfo_depth_oncc: Arc::from(Vec::<(u8, f32)>::new()),
                        });

                        spawner_params_list[index].push(spawner_params.clone());
                    }
                }
            }

            let new = SoundfontInstrument {
                bank: preset.bank as u8,
                preset: preset.preset as u8,
                spawner_params_list,
                release_spawner_params_list,
                rr_state: (0..128 * 128).map(|_| None).collect(),
            };
            instruments.push(new);
        }

        Ok(SampleSoundfont {
            instruments,
            stream_params,
            // SF2 isn't used by schwung-sfz today; pick a safe default
            // that disables auto-gain (1.0 = no attenuation, plugin
            // computes preset_attenuation = 1.0 below threshold).
            estimated_voice_peak: 0.5,
        })
    }

    /// MOVE FORK / 2026-05-17: max absolute f32 amplitude any single
    /// voice can produce in this preset (sample_peak × region.volume).
    /// The plugin uses this to derive a per-preset attenuation so hot
    /// SFZ libraries don't dominate user mixing.
    pub fn estimated_voice_peak(&self) -> f32 {
        self.estimated_voice_peak
    }
}

impl std::fmt::Debug for SampleSoundfont {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "SampleSoundfont")
    }
}

impl SoundfontBase for SampleSoundfont {
    fn stream_params(&self) -> &'_ AudioStreamParams {
        &self.stream_params
    }

    fn get_attack_voice_spawners_at(
        &self,
        bank: u8,
        preset: u8,
        key: u8,
        vel: u8,
    ) -> Vec<Box<dyn VoiceSpawner>> {
        use simdeez::*; // nuts

        use simdeez::prelude::*;

        simd_runtime_generate!(
            fn get(
                key: u8,
                vel: u8,
                sf: &SoundfontInstrument,
                stream_params: &AudioStreamParams,
            ) -> Vec<Box<dyn VoiceSpawner>> {
                if sf.spawner_params_list.is_empty() {
                    return Vec::new();
                }

                let index = key_vel_to_index(key, vel);
                // MOVE FORK: round-robin position for this NoteOn. If
                // rr_state[index] is Some, advance the counter; spawners
                // with `seq_position` matching (counter % seq_length + 1)
                // fire, others skip. seq_position==0 is "always fire".
                let rr_position: Option<u8> = sf
                    .rr_state
                    .get(index)
                    .and_then(|opt| opt.as_ref())
                    .map(|rr| {
                        let prev = rr
                            .counter
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        (prev % rr.seq_length as u32) as u8 + 1
                    });

                let mut vec = Vec::<Box<dyn VoiceSpawner>>::new();
                for spawner in &sf.spawner_params_list[index] {
                    if let Some(pos) = rr_position {
                        if spawner.seq_position != 0 && spawner.seq_position != pos {
                            continue;
                        }
                    }
                    match stream_params.channels {
                        ChannelCount::Stereo => vec.push(Box::new(
                            StereoSampledVoiceSpawner::<S>::new(spawner, vel, *stream_params),
                        )),
                        ChannelCount::Mono => vec.push(Box::new(
                            MonoSampledVoiceSpawner::<S>::new(spawner, vel, *stream_params),
                        )),
                    }
                }
                vec
            }
        );

        let empty = SoundfontInstrument {
            bank: 0,
            preset: 0,
            spawner_params_list: Vec::new(),
            release_spawner_params_list: Vec::new(),
            rr_state: Vec::new(),
        };

        let instrument = self
            .instruments
            .iter()
            .find(|i| i.bank == bank && i.preset == preset)
            .unwrap_or(&empty);

        get(key, vel, instrument, self.stream_params())
    }

    fn get_release_voice_spawners_at(
        &self,
        bank: u8,
        preset: u8,
        key: u8,
        vel: u8,
    ) -> Vec<Box<dyn VoiceSpawner>> {
        use simdeez::*;
        use simdeez::prelude::*;

        simd_runtime_generate!(
            fn get(
                key: u8,
                vel: u8,
                sf: &SoundfontInstrument,
                stream_params: &AudioStreamParams,
            ) -> Vec<Box<dyn VoiceSpawner>> {
                if sf.release_spawner_params_list.is_empty() {
                    return Vec::new();
                }

                let index = key_vel_to_index(key, vel);
                let mut vec = Vec::<Box<dyn VoiceSpawner>>::new();
                for spawner in &sf.release_spawner_params_list[index] {
                    match stream_params.channels {
                        ChannelCount::Stereo => vec.push(Box::new(
                            StereoSampledVoiceSpawner::<S>::new(spawner, vel, *stream_params),
                        )),
                        ChannelCount::Mono => vec.push(Box::new(
                            MonoSampledVoiceSpawner::<S>::new(spawner, vel, *stream_params),
                        )),
                    }
                }
                vec
            }
        );

        let empty = SoundfontInstrument {
            bank: 0,
            preset: 0,
            spawner_params_list: Vec::new(),
            release_spawner_params_list: Vec::new(),
            rr_state: Vec::new(),
        };

        let instrument = self
            .instruments
            .iter()
            .find(|i| i.bank == bank && i.preset == preset)
            .unwrap_or(&empty);

        get(key, vel, instrument, self.stream_params())
    }
}
