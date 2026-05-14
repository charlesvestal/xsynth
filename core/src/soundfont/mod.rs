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

use self::audio::load_audio_file;
pub use self::audio::AudioLoadError;

use super::{
    voice::VoiceControlData,
    voice::{EnvelopeParameters, Voice},
};
use crate::{helpers::db_to_amp, voice::{CcState, EnvelopeDescriptor}, AudioStreamParams, ChannelCount};

pub use xsynth_soundfonts::{sf2::Sf2ParseError, sfz::SfzParseError};

mod audio;
mod config;
mod sample_storage;
mod utils;
mod voice_spawners;
use utils::*;
use voice_spawners::*;
pub use sample_storage::{MmapHolder, SampleStorage};

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
    sample: Arc<[Arc<SampleStorage>]>,
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
        let samples: Result<HashMap<_, _>, _> = unique_sample_params
            .into_iter()
            .map(|params| -> Result<(_, _), LoadSfzError> {
                check_cancel()?;
                let sample = load_audio_file(&params.path, stream_params)?;
                Ok((params, sample))
            })
            .collect();
        let samples = samples?;

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
                            * cents_factor(region.tune as f32);

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
                    let vol_mult = (vol_vel / 127.0).powi(2);
                    let vol_db_add =
                        (key as f32 - region.amp_keycenter as f32) * region.amp_keytrack;
                    let vol_db = (region.volume as f32 + vol_db_add).clamp(-96.0, 12.0);
                    let volume = vol_mult * db_to_amp(vol_db);

                    let sample_rate = samples[&params].1;

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

                    let mut region_samples = samples[&params].0.clone();
                    if stream_params.channels == ChannelCount::Stereo && region_samples.len() == 1 {
                        region_samples =
                            Arc::new([region_samples[0].clone(), region_samples[0].clone()]);
                    }

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

        Ok(SampleSoundfont {
            instruments: vec![SoundfontInstrument {
                bank: options.bank.unwrap_or(0),
                preset: options.preset.unwrap_or(0),
                spawner_params_list,
                release_spawner_params_list,
                rr_state,
            }],
            stream_params,
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
                            sample: sample_storage,
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
        })
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
