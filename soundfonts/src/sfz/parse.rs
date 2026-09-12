use std::{
    collections::HashMap,
    ops::RangeInclusive,
    path::{Path, PathBuf},
};

use crate::{FilterType, LoopMode};
use super::region::TriggerType;

use self::defines::apply_defines;

use super::grammar::{ErrorTolerantToken, Group, Opcode, Token, TokenKind};
use regex_bnf::{FileLocation, ParseError};
use thiserror::Error;

mod defines;
mod resolver;
use resolver::TokenResolver;

#[derive(Debug, Clone)]
pub enum SfzOpcode {
    Lovel(u8),
    Hivel(u8),
    Key(i8),
    Lokey(i8),
    Hikey(i8),
    PitchKeycenter(i8),
    Volume(i16),
    Pan(i8),
    Sample(String),
    LoopMode(LoopMode),
    LoopStart(u32),
    LoopEnd(u32),
    /// MOVE FORK / Phase 8: SFZ `loop_crossfade=<seconds>` — fade
    /// duration in seconds applied across the loop point so the
    /// transition from `loop_end` back to `loop_start` is smooth.
    /// Stored on the region as raw f32 seconds; converted to frame
    /// count at voice spawn (rate * seconds).
    LoopCrossfade(f32),
    /// MOVE FORK / Phase 11: SFZ `amplfo_freq` (Hz) + `amplfo_depth`
    /// (dB) — sine LFO modulating the voice's amp.
    AmpLfoFreq(f32),
    AmpLfoDepth(f32),
    /// MOVE FORK / Phase 11: live CC modulation of amp LFO freq/depth.
    /// `amplfo_freq_oncc<N>=<Hz delta>` — at CC=127, freq becomes
    /// base + delta. `amplfo_depth_oncc<N>=<dB delta>` — same for depth.
    AmpLfoFreqOncc(u8, f32),
    AmpLfoDepthOncc(u8, f32),
    /// MOVE FORK / Phase 11: SFZ `fillfo_freq` (Hz) + `fillfo_depth`
    /// (cents) — sine LFO modulating the voice's filter cutoff.
    FilLfoFreq(f32),
    FilLfoDepth(f32),
    FilLfoFreqOncc(u8, f32),
    FilLfoDepthOncc(u8, f32),
    /// MOVE FORK / Phase 11: SFZ `panlfo_freq` (Hz) + `panlfo_depth`
    /// (-100..100 = full L/R sweep). Sine LFO modulating voice pan.
    PanLfoFreq(f32),
    PanLfoDepth(f32),
    PanLfoFreqOncc(u8, f32),
    PanLfoDepthOncc(u8, f32),
    /// MOVE FORK / Phase 11: filter envelope (autowah). ADSR routed
    /// into cutoff cents — `fileg_depth` is the full-swing cents added
    /// at envelope peak.
    FilegAttack(f32),
    FilegDecay(f32),
    FilegSustain(f32),
    FilegRelease(f32),
    FilegDepth(f32),
    /// MOVE FORK / 2026-05-16: optional curve table reference for the
    /// filter envelope. Without it, fileg adds `level * depth` cents
    /// per sample (exponential cutoff sweep). With it, the
    /// LiveCutoffState looks up `curve[level * 127]` as a 0..1
    /// multiplier on depth, so the converter can pre-bake a linear-
    /// Hz mapping (DS's `<envelope translation="linear">` semantic)
    /// into an exponential-cents domain.
    FilegCurve(u8),
    /// MOVE FORK / Phase 11: pitch LFO (vibrato). depth in cents.
    PitchLfoFreq(f32),
    PitchLfoDepth(f32),
    PitchLfoFreqOncc(u8, f32),
    PitchLfoDepthOncc(u8, f32),
    Offset(u32),
    Cutoff(f32),
    Resonance(f32),
    /// MOVE FORK / 2026-05-16: SFZ keyswitch opcodes. `sw_default` is the
    /// initial keyswitch key for the file; `sw_last` constrains a region
    /// to fire only when the most-recent keyswitch press matches this key.
    /// `sw_lokey`/`sw_hikey` define the keyswitch range — keys in that
    /// range update the "last keyswitch" state instead of triggering
    /// regions. v1 implementation: regions whose sw_last doesn't match
    /// sw_default are dropped at preset-build time (no live keyswitching).
    /// Lets Salamander Grand Piano load without doubling Natural+Retuned
    /// voices.
    SwDefault(i8),
    SwLast(i8),
    SwLokey(i8),
    SwHikey(i8),
    AmpKeycenter(i8),
    AmpKeytrack(f32),
    AmpVeltrack(f32),
    PanKeycenter(i8),
    PanKeytrack(f32),
    PanVeltrack(f32),
    FilVeltrack(i16),
    FilKeycenter(i8),
    FilKeytrack(i16),
    FilterType(FilterType),
    DefaultPath(String),
    Tune(i16),
    AmpegEnvelope(SfzAmpegEnvelope),
    Trigger(TriggerType),
    /// MOVE FORK / 2026-05-19: SFZ `polyphony=N` opcode. Max
    /// simultaneous voices for the enclosing scope. We collect the
    /// minimum value across all regions of a soundfont as the author's
    /// declared cap.
    Polyphony(u32),
    /// MOVE FORK / 2026-05-19: SFZ `group=N` opcode — this region
    /// belongs to choke group N. Distinct from the `<group>` SECTION
    /// header (SfzGroupType::Group) — same word, different concept in
    /// SFZ spec.
    SfzGroup(u32),
    /// MOVE FORK / 2026-05-19: SFZ `off_by=N` opcode — this region's
    /// note-on chokes voices currently playing from group N.
    SfzOffBy(u32),
    SeqLength(u32),
    SeqPosition(u32),
    /// MOVE FORK: ARIA `set_cc<N>=v` or `set_hdcc<N>=v`. Stores the
    /// channel's initial CC<N> value (normalized 0..1) into the parser's
    /// CC state for later use by `_oncc` modulator opcodes.
    AriaCcInit(u8, f32),
    /// MOVE FORK: ARIA `<base>_oncc<N>=v` modulator. parse_sf_root
    /// resolves it by looking up CC<N>'s init value and adding the
    /// scaled contribution `v * cc_init` to the matching base opcode.
    /// Currently limited to ampeg_* — covers the case where a serious
    /// SFZ library relies on a CC72-style release knob (Splendid Grand
    /// Piano: `ampeg_release_oncc72=2` with `set_hdcc72=0.35` baked as
    /// `ampeg_release += 0.7`). Without this support those presets
    /// load with the default 10 ms release and sound abrupt.
    AriaOncc {
        base: AriaOnccBase,
        cc: u8,
        value: f32,
    },
    /// MOVE FORK: `locc<N>=<v>` — region/group only fires when CC<N> ≥ v.
    /// Statically evaluated against the parser's ARIA CC state at region
    /// build time. (No live CC modulation yet; regions that don't pass
    /// at the default CC state are dropped entirely.)
    LoCc(u8, u8),
    /// MOVE FORK: `hicc<N>=<v>` — region/group only fires when CC<N> ≤ v.
    HiCc(u8, u8),
    /// MOVE FORK: `volume_oncc<N>=<dB>` — live volume modulation. Stored
    /// on the region (not folded into a static value at load time) so a
    /// future SIMD generator can sample CC<N> each block and apply
    /// `db_to_amp(delta·cc/127)`. Multiple opcodes for the same region
    /// accumulate; duplicate CCs follow SFZ "last assignment wins".
    VolumeOncc(u8, f32),
    /// MOVE FORK: `cutoff_oncc<N>=<cents>` — live filter cutoff modulation.
    /// Cents offset applied to the base cutoff frequency: at CC=N, the
    /// per-voice filter coefficients are recomputed for
    /// `cutoff * 2^(cents·cc/127/1200)`.
    CutoffOncc(u8, f32),
    /// MOVE FORK: `resonance_oncc<N>=<dB>` — live filter resonance
    /// modulation. dB offset added to the base resonance at CC=N.
    ResonanceOncc(u8, f32),
    /// MOVE FORK: `pan_oncc<N>=<percent>` — live pan modulation. Pan
    /// offset (-100..100) added to the base pan at CC=N.
    PanOncc(u8, f32),
    /// MOVE FORK: `<base>_curvecc<N>=<curve_id>` — references a curve
    /// table that shapes the CC lookup for this binding. Base is one of
    /// volume/cutoff/resonance/pan; the matching `_oncc<N>` opcode
    /// provides the full-swing magnitude. SIMD generators use
    /// `curve[cc]` instead of `cc/127` when a curve is referenced.
    VolumeCurvecc(u8, u8),
    CutoffCurvecc(u8, u8),
    ResonanceCurvecc(u8, u8),
    PanCurvecc(u8, u8),
    /// MOVE FORK / 2026-05-19: `tune_oncc<N>=<cents>` — live pitch
    /// modulation. cents is the value added at CC=full-scale; runtime
    /// scales linearly by cc/127. Drives DS PITCH / GROUP_TUNING knobs.
    TuneOncc(u8, f32),
    /// MOVE FORK: `index=<id>` inside a `<curve>` block. Identifies the
    /// curve table being defined.
    CurveIndex(u8),
    /// MOVE FORK: `v<NNN>=<f32>` inside a `<curve>` block. NNN is
    /// 0..127. Defines one sample point of the current curve.
    CurvePoint(u8, f32),
    /// MOVE FORK / 2026-05-19: CC-based amplitude crossfade — `xfin_locc<N>`
    /// (region silent below this CC value, ramping up by hi), `xfin_hicc<N>`
    /// (region full above this CC value), and matching `xfout_*` for the
    /// fade-out side. Each (lo, hi) pair defines a linear (or equal-power
    /// with `xf_cccurve=power`) amplitude scale evaluated STATICALLY at
    /// region build time against the parser's `set_cc` state. xsynth has
    /// no live xfin/xfout sweep yet — regions whose static factor is
    /// effectively zero are dropped; otherwise the factor is folded into
    /// `volume=` as dB attenuation. Drives velocity-layer SFZ libraries
    /// like Pianobook Fake Dulcimer (Tremolo) where CC1 picks which
    /// velocity-layer group fires.
    XfInLoCc(u8, u8),
    XfInHiCc(u8, u8),
    XfOutLoCc(u8, u8),
    XfOutHiCc(u8, u8),
    /// MOVE FORK / 2026-05-19: `xf_cccurve` — "gain" (linear-amp, default)
    /// or "power" (equal-power, sin/cos curve). Applied to xfin/xfout
    /// CC-based crossfades at build time.
    XfCcCurve(XfCurve),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum XfCurve {
    #[default]
    Gain,
    Power,
}

#[derive(Debug, Clone, Copy)]
pub enum AriaOnccBase {
    AmpegAttack,
    AmpegHold,
    AmpegDecay,
    AmpegSustain,
    AmpegRelease,
    AmpegDelay,
    AmpegStart,
}

#[derive(Debug, Clone)]
#[allow(clippy::enum_variant_names)]
pub enum SfzAmpegEnvelope {
    AmpegStart(f32),
    AmpegDelay(f32),
    AmpegAttack(f32),
    AmpegHold(f32),
    AmpegDecay(f32),
    AmpegSustain(f32),
    AmpegRelease(f32),
    AmpegVel2Release(f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfzGroupType {
    Region,
    Group,
    Master,
    Global,
    Control,
    /// MOVE FORK: SFZv2 `<curve>` block — defines a 128-point curve
    /// referenced by `<base>_curvecc<N>=<curve_id>` opcodes. Curve
    /// shapes the `cc/127` lookup that `_oncc` modulators apply.
    /// The block body uses `index=<id>` plus `vNNN=<value>` opcodes
    /// (0..127) to define the table.
    Curve,
    Other,
}

#[derive(Debug, Clone)]
pub enum SfzToken {
    Group(SfzGroupType),
    Opcode(SfzOpcode),
}

#[derive(Debug, Clone)]
pub enum SfzTokenWithMeta {
    Group(SfzGroupType),
    Opcode(SfzOpcode),
    Import(String),
    Define(String, String),
}

/// Parameters of an error generated while validating an SFZ file.
#[derive(Error, Debug, Clone)]
pub struct SfzValidationError {
    pub pos: FileLocation,
    pub message: String,
}

impl SfzValidationError {
    #[allow(dead_code)]
    pub(super) fn new(pos: FileLocation, message: String) -> Self {
        Self { pos, message }
    }
}

impl std::fmt::Display for SfzValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at {}", self.message, self.pos)
    }
}

/// Errors that can be generated when parsing an SFZ file.
#[derive(Error, Debug, Clone)]
pub enum SfzParseError {
    #[error("Failed to parse SFZ file: {0}")]
    GrammarError(#[from] ParseError),

    #[error("Failed to parse SFZ file: {0}")]
    ValidationError(#[from] SfzValidationError),

    #[error("Failed to read file: {0}")]
    FailedToReadFile(PathBuf),

    #[error("Cyclic include detected while reading: {0}")]
    IncludeCycle(PathBuf),
}

fn parse_key_number(val: &str) -> Option<i8> {
    match val.parse::<i8>().ok() {
        Some(val) => Some(val.clamp(-1, 127)),
        None => {
            let note: String = val
                .chars()
                .filter(|c| !(c.is_ascii_digit() || c == &'-'))
                .collect();
            let semitone: i16 = match note.to_lowercase().as_str() {
                "c" => 0,
                "c#" => 1,
                "db" => 1,
                "d" => 2,
                "d#" => 3,
                "eb" => 3,
                "e" => 4,
                "f" => 5,
                "f#" => 6,
                "gb" => 6,
                "g" => 7,
                "g#" => 8,
                "ab" => 8,
                "a" => 9,
                "a#" => 10,
                "bb" => 10,
                "b" => 11,
                _ => return None,
            };
            let octave: String = val
                .chars()
                .filter(|c| c.is_ascii_digit() || c == &'-')
                .collect();
            let octave: i16 = octave.parse().ok().unwrap_or(-10);
            if octave < -1 {
                None
            } else {
                let midi_note = 12 + semitone + octave * 12;
                Some(midi_note.clamp(-1, 127) as i8)
            }
        }
    }
}

fn parse_u8_in_range(val: &str, range: RangeInclusive<u8>) -> Option<u8> {
    val.parse()
        .ok()
        .map(|val: u8| val.clamp(*range.start(), *range.end()))
}

fn parse_i8_in_range(val: &str, range: RangeInclusive<i8>) -> Option<i8> {
    val.parse()
        .ok()
        .map(|val: i8| val.clamp(*range.start(), *range.end()))
}

fn parse_i16_in_range(val: &str, range: RangeInclusive<i16>) -> Option<i16> {
    val.parse()
        .ok()
        .map(|val: i16| val.clamp(*range.start(), *range.end()))
}

fn parse_u32_in_range(val: &str, range: RangeInclusive<u32>) -> Option<u32> {
    val.parse()
        .ok()
        .map(|val: u32| val.clamp(*range.start(), *range.end()))
}

fn parse_float_in_range(val: &str, range: RangeInclusive<f32>) -> Option<f32> {
    val.parse()
        .ok()
        .map(|val: f32| val.clamp(*range.start(), *range.end()))
}

fn parse_filter_kind(val: &str) -> Option<FilterType> {
    match val {
        "lpf_1p" => Some(FilterType::LowPassPole),
        "lpf_2p" => Some(FilterType::LowPass),
        "lpf_4p" => Some(FilterType::LowPass),
        "lpf_6p" => Some(FilterType::LowPass),
        "hpf_1p" => Some(FilterType::HighPass),
        "hpf_2p" => Some(FilterType::HighPass),
        "hpf_4p" => Some(FilterType::HighPass),
        "hpf_6p" => Some(FilterType::HighPass),
        "bpf_1p" => Some(FilterType::BandPass),
        "bpf_2p" => Some(FilterType::BandPass),
        // MOVE FORK / 2026-05-19: notch / band-reject filter. SFZ spec
        // uses `brf_2p`; xsynth maps to biquad's Type::Notch.
        "brf_1p" | "brf_2p" => Some(FilterType::Notch),
        _ => None,
    }
}

fn parse_trigger(val: &str) -> Option<TriggerType> {
    match val {
        "attack" => Some(TriggerType::Attack),
        "release" => Some(TriggerType::Release),
        // first / legato exist in the SFZ spec but xsynth doesn't model
        // those triggers; default to Attack so they behave as standard
        // note-on regions rather than silently disappear.
        "first" | "legato" => Some(TriggerType::Attack),
        _ => None,
    }
}

fn parse_loop_mode(val: &str) -> Option<LoopMode> {
    match val {
        "no_loop" => Some(LoopMode::NoLoop),
        "one_shot" => Some(LoopMode::OneShot),
        "loop_continuous" => Some(LoopMode::LoopContinuous),
        "loop_sustain" => Some(LoopMode::LoopSustain),
        _ => None,
    }
}

fn parse_sfz_opcode(
    opcode: Opcode,
    defines: &HashMap<String, String>,
) -> Result<Option<SfzOpcode>, SfzValidationError> {
    let opcode_value = opcode.value.as_string();
    let name = apply_defines(opcode.name.name.text, defines);
    let val = apply_defines(&opcode_value, defines);

    use SfzAmpegEnvelope::*;
    use SfzOpcode::*;

    let val = val.as_ref();
    let name = name.as_ref();

    // MOVE FORK: ARIA `set_cc<N>` / `set_hdcc<N>` — control-block CC
    // initializer. parse_sf_root collects these to bake `_oncc` values.
    if let Some(rest) = name.strip_prefix("set_hdcc") {
        if let Ok(n) = rest.parse::<u8>() {
            if let Ok(v) = val.parse::<f32>() {
                return Ok(Some(AriaCcInit(n, v.clamp(0.0, 1.0))));
            }
        }
        return Ok(None);
    }
    if let Some(rest) = name.strip_prefix("set_cc") {
        if let Ok(n) = rest.parse::<u8>() {
            if let Ok(v) = val.parse::<u8>() {
                return Ok(Some(AriaCcInit(n, (v as f32 / 127.0).clamp(0.0, 1.0))));
            }
        }
        return Ok(None);
    }

    // MOVE FORK / 2026-05-19: xfin_locc<N> / xfin_hicc<N> / xfout_locc<N>
    // / xfout_hicc<N> — CC-based amplitude crossfade. Match BEFORE the
    // generic `locc`/`hicc` prefixes since those would otherwise eat the
    // "xfin_locc..." suffix. Velocity variants (xfin_lovel/etc.) are not
    // parsed here yet — they need per-voice (not per-region) application.
    if let Some(rest) = name.strip_prefix("xfin_locc") {
        if let Ok(n) = rest.parse::<u8>() {
            if let Ok(v) = val.parse::<u8>() {
                return Ok(Some(XfInLoCc(n, v.min(127))));
            }
        }
        return Ok(None);
    }
    if let Some(rest) = name.strip_prefix("xfin_hicc") {
        if let Ok(n) = rest.parse::<u8>() {
            if let Ok(v) = val.parse::<u8>() {
                return Ok(Some(XfInHiCc(n, v.min(127))));
            }
        }
        return Ok(None);
    }
    if let Some(rest) = name.strip_prefix("xfout_locc") {
        if let Ok(n) = rest.parse::<u8>() {
            if let Ok(v) = val.parse::<u8>() {
                return Ok(Some(XfOutLoCc(n, v.min(127))));
            }
        }
        return Ok(None);
    }
    if let Some(rest) = name.strip_prefix("xfout_hicc") {
        if let Ok(n) = rest.parse::<u8>() {
            if let Ok(v) = val.parse::<u8>() {
                return Ok(Some(XfOutHiCc(n, v.min(127))));
            }
        }
        return Ok(None);
    }

    // MOVE FORK: locc<N>=v / hicc<N>=v — static CC-range gating for
    // regions/groups. Parsed here, resolved against the parser's
    // cc_state at region build time.
    if let Some(rest) = name.strip_prefix("locc") {
        if let Ok(n) = rest.parse::<u8>() {
            if let Ok(v) = val.parse::<u8>() {
                return Ok(Some(LoCc(n, v.min(127))));
            }
        }
        return Ok(None);
    }
    if let Some(rest) = name.strip_prefix("hicc") {
        if let Ok(n) = rest.parse::<u8>() {
            if let Ok(v) = val.parse::<u8>() {
                return Ok(Some(HiCc(n, v.min(127))));
            }
        }
        return Ok(None);
    }

    // MOVE FORK: ARIA `<base>_oncc<N>` modulator.
    //  - ampeg_*: parse_sf_root resolves the CC value and adds the
    //    scaled contribution to the base opcode at load time (static
    //    bake — knob position frozen at load).
    //  - volume: stored as a live binding on the region; a future SIMD
    //    generator can sample CC<N> each block.
    //  - others: silently dropped (matches xsynth's previous behavior).
    // `_curvecc` variants also silently fall through (no curve support).
    if let Some(idx) = name.find("_oncc") {
        let base_name = &name[..idx];
        let cc_part = &name[idx + "_oncc".len()..];
        if let Ok(cc_n) = cc_part.parse::<u8>() {
            let ampeg_base = match base_name {
                "ampeg_attack" => Some(AriaOnccBase::AmpegAttack),
                "ampeg_hold" => Some(AriaOnccBase::AmpegHold),
                "ampeg_decay" => Some(AriaOnccBase::AmpegDecay),
                "ampeg_sustain" => Some(AriaOnccBase::AmpegSustain),
                "ampeg_release" | "ampeg_releasecc" => Some(AriaOnccBase::AmpegRelease),
                "ampeg_delay" => Some(AriaOnccBase::AmpegDelay),
                "ampeg_start" => Some(AriaOnccBase::AmpegStart),
                _ => None,
            };
            // MOVE FORK / 2026-09-12: ARIA extended CCs. `_oncc<N>` accepts
            // N up to 255 (Splendid Grand Piano ships
            // `ampeg_decay_oncc133`), but `CcState` is 128 atomics and no
            // MIDI message can ever set CC >= 128 — so a LIVE binding on
            // one is both unreachable and an out-of-bounds index at voice
            // spawn. Drop those here.
            //
            // `ampeg_*` is the exception and must NOT be dropped: it is
            // baked statically against the control block's `set_hdcc<N>`
            // in `parse_sf_root`, which is the whole reason that piano's
            // envelopes come out right. `region.rs` keeps the fold and
            // skips only the runtime half. Dropping the opcode here
            // instead would load and play — with a silently wrong decay.
            if cc_n >= 128 && ampeg_base.is_none() {
                return Ok(None);
            }
            // Live binding: stored on the region, sampled at render time.
            if base_name == "volume" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(VolumeOncc(cc_n, v)));
                }
                return Ok(None);
            }
            if base_name == "cutoff" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(CutoffOncc(cc_n, v)));
                }
                return Ok(None);
            }
            if base_name == "resonance" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(ResonanceOncc(cc_n, v)));
                }
                return Ok(None);
            }
            if base_name == "pan" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(PanOncc(cc_n, v)));
                }
                return Ok(None);
            }
            // MOVE FORK / 2026-05-19: live tune_oncc<N>=<cents>. Drives
            // DS PITCH / GROUP_TUNING knob bindings — value is cents
            // added at full-CC (CC=127 contributes the full value).
            if base_name == "tune" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(TuneOncc(cc_n, v)));
                }
                return Ok(None);
            }
            if base_name == "amplfo_freq" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(AmpLfoFreqOncc(cc_n, v)));
                }
                return Ok(None);
            }
            if base_name == "amplfo_depth" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(AmpLfoDepthOncc(cc_n, v)));
                }
                return Ok(None);
            }
            if base_name == "fillfo_freq" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(FilLfoFreqOncc(cc_n, v)));
                }
                return Ok(None);
            }
            if base_name == "fillfo_depth" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(FilLfoDepthOncc(cc_n, v)));
                }
                return Ok(None);
            }
            if base_name == "panlfo_freq" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(PanLfoFreqOncc(cc_n, v)));
                }
                return Ok(None);
            }
            if base_name == "panlfo_depth" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(PanLfoDepthOncc(cc_n, v)));
                }
                return Ok(None);
            }
            if base_name == "pitchlfo_freq" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(PitchLfoFreqOncc(cc_n, v)));
                }
                return Ok(None);
            }
            if base_name == "pitchlfo_depth" {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(PitchLfoDepthOncc(cc_n, v)));
                }
                return Ok(None);
            }
            if let Some(base) = ampeg_base {
                if let Ok(v) = val.parse::<f32>() {
                    return Ok(Some(AriaOncc { base, cc: cc_n, value: v }));
                }
            }
        }
        // unrecognized _oncc: silently drop
        return Ok(None);
    }
    // MOVE FORK: `<base>_curvecc<N>=<curve_id>` for volume/cutoff/
    // resonance/pan bases. The matching `_oncc<N>` provides the
    // full-swing magnitude; the curve shapes the CC lookup.
    if let Some(idx) = name.find("_curvecc") {
        let base_name = &name[..idx];
        let cc_part = &name[idx + "_curvecc".len()..];
        if let Ok(cc_n) = cc_part.parse::<u8>() {
            if cc_n >= 128 {
                return Ok(None);
            }
            if let Ok(curve_id) = val.parse::<u8>() {
                match base_name {
                    "volume" => return Ok(Some(VolumeCurvecc(cc_n, curve_id))),
                    "cutoff" => return Ok(Some(CutoffCurvecc(cc_n, curve_id))),
                    "resonance" => return Ok(Some(ResonanceCurvecc(cc_n, curve_id))),
                    "pan" => return Ok(Some(PanCurvecc(cc_n, curve_id))),
                    _ => return Ok(None),
                }
            }
        }
        return Ok(None);
    }

    // MOVE FORK: <curve> block opcodes.
    if name == "index" {
        if let Ok(id) = val.parse::<u8>() {
            return Ok(Some(CurveIndex(id)));
        }
        return Ok(None);
    }
    if name.len() >= 2 && name.starts_with('v') {
        let nstr = &name[1..];
        if nstr.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(n) = nstr.parse::<u8>() {
                if n <= 127 {
                    if let Ok(v) = val.parse::<f32>() {
                        return Ok(Some(CurvePoint(n, v)));
                    }
                }
            }
        }
    }

    Ok(match name {
        "lokey" => parse_key_number(val).map(Lokey),
        "hikey" => parse_key_number(val).map(Hikey),
        "lovel" => parse_u8_in_range(val, 0..=128).map(Lovel),
        "hivel" => parse_u8_in_range(val, 0..=128).map(Hivel),
        "volume" => parse_i16_in_range(val, -144..=6).map(Volume),
        "pan" => parse_i8_in_range(val, -100..=100).map(Pan),
        "pitch_keycenter" => parse_key_number(val).map(PitchKeycenter),
        "key" => parse_key_number(val).map(Key),
        // MOVE FORK / 2026-05-16: keyswitch opcodes (sw_*).
        "sw_default" => parse_key_number(val).map(SwDefault),
        "sw_last"    => parse_key_number(val).map(SwLast),
        "sw_lokey"   => parse_key_number(val).map(SwLokey),
        "sw_hikey"   => parse_key_number(val).map(SwHikey),
        "cutoff" => parse_float_in_range(val, 1.0..=100000.0).map(Cutoff),
        "resonance" => parse_float_in_range(val, 0.0..=40.0).map(Resonance),
        "amp_keycenter" => parse_key_number(val).map(AmpKeycenter),
        "amp_keytrack" => parse_float_in_range(val, -96.0..=12.0).map(AmpKeytrack),
        "amp_veltrack" => parse_float_in_range(val, -100.0..=100.0).map(AmpVeltrack),
        "pan_keycenter" => parse_key_number(val).map(PanKeycenter),
        "pan_keytrack" => parse_float_in_range(val, -100.0..=100.0).map(PanKeytrack),
        "pan_veltrack" => parse_float_in_range(val, -100.0..=100.0).map(PanVeltrack),
        "fil_veltrack" => parse_i16_in_range(val, -9600..=9600).map(FilVeltrack),
        "fil_keytrack" => parse_i16_in_range(val, 0..=1200).map(FilKeytrack),
        "fil_keycenter" => parse_key_number(val).map(FilKeycenter),
        "fil_type" => parse_filter_kind(val).map(FilterType),
        "loop_mode" | "loopmode" => parse_loop_mode(val).map(LoopMode),
        "loop_start" | "loopstart" => parse_u32_in_range(val, 0..=u32::MAX).map(LoopStart),
        "loop_end" | "loopend" => parse_u32_in_range(val, 0..=u32::MAX).map(LoopEnd),
        "loop_crossfade" | "loopcrossfade" => val.parse::<f32>().ok().map(LoopCrossfade),
        "amplfo_freq" => val.parse::<f32>().ok().map(AmpLfoFreq),
        "amplfo_depth" => val.parse::<f32>().ok().map(AmpLfoDepth),
        "fillfo_freq" => val.parse::<f32>().ok().map(FilLfoFreq),
        "fillfo_depth" => val.parse::<f32>().ok().map(FilLfoDepth),
        "panlfo_freq" => val.parse::<f32>().ok().map(PanLfoFreq),
        "panlfo_depth" => val.parse::<f32>().ok().map(PanLfoDepth),
        "fileg_attack"  => val.parse::<f32>().ok().map(FilegAttack),
        "fileg_decay"   => val.parse::<f32>().ok().map(FilegDecay),
        "fileg_sustain" => val.parse::<f32>().ok().map(FilegSustain),
        "fileg_release" => val.parse::<f32>().ok().map(FilegRelease),
        "fileg_depth"   => val.parse::<f32>().ok().map(FilegDepth),
        "fileg_curve"   => val.parse::<u8>().ok().map(FilegCurve),
        "pitchlfo_freq"  => val.parse::<f32>().ok().map(PitchLfoFreq),
        "pitchlfo_depth" => val.parse::<f32>().ok().map(PitchLfoDepth),
        "offset" => parse_u32_in_range(val, 0..=u32::MAX).map(Offset),
        // MOVE FORK / 2026-05-17: `prefix_sfz_path` is an alternate
        // spelling some authors (jRhodes3d, others) use for the same
        // semantic — prepend this string to every region's `sample=`
        // path. Without it, jRhodes3d's outer .sfz files reference
        // samples in a sibling subdir and we emit silence.
        "default_path" | "prefix_sfz_path"
            => Some(DefaultPath(val.replace('\\', "/"))),
        "tune" => parse_i16_in_range(val, -2400..=2400).map(Tune),
        // MOVE FORK / 2026-05-19: SFZ `polyphony=N` opcode. Declares the
        // maximum simultaneous voices for the enclosing scope. Stored on
        // the region (and inherited from group/master/global like other
        // opcodes); xsynth-core's SampleSoundfont surfaces the minimum
        // across all regions as the "preset author's polyphony intent",
        // which the C plugin uses to override the auto-heuristic when
        // the author explicitly set a monophonic (or other narrow) cap.
        "polyphony" => parse_u32_in_range(val, 1..=128).map(Polyphony),
        // MOVE FORK / 2026-05-19: SFZ choke groups.
        //   group=N    — this region belongs to group N.
        //   off_by=N   — this region's note-on chokes group-N voices.
        // Maps to xsynth-core's exclusive_class field (existing SF2
        // chokeGroup mechanism). Both opcodes accept 0..=4_294_967_295
        // in spec but we cap at u8 — the SF2 underlying field is u8 —
        // and skip group=0 (the "no group" sentinel in SFZ).
        "group" => parse_u32_in_range(val, 0..=u32::MAX).map(SfzGroup),
        "off_by" | "offBy" => parse_u32_in_range(val, 0..=u32::MAX).map(SfzOffBy),
        "trigger" => parse_trigger(val).map(Trigger),
        "seq_length" | "seqlength" => parse_u32_in_range(val, 0..=255).map(SeqLength),
        "seq_position" | "seqposition" => parse_u32_in_range(val, 0..=255).map(SeqPosition),
        // MOVE FORK / 2026-05-19: xf_cccurve = gain | power. Defaults to
        // gain (linear-amp). Power = equal-power sin/cos curve.
        "xf_cccurve" => match val {
            "gain"  => Some(XfCcCurve(XfCurve::Gain)),
            "power" => Some(XfCcCurve(XfCurve::Power)),
            _ => None,
        },

        "ampeg_delay" => parse_float_in_range(val, 0.0..=100.0)
            .map(AmpegDelay)
            .map(AmpegEnvelope),
        "ampeg_start" => parse_float_in_range(val, 0.0..=100.0)
            .map(AmpegStart)
            .map(AmpegEnvelope),
        "ampeg_attack" => parse_float_in_range(val, 0.0..=100.0)
            .map(AmpegAttack)
            .map(AmpegEnvelope),
        "ampeg_hold" => parse_float_in_range(val, 0.0..=100.0)
            .map(AmpegHold)
            .map(AmpegEnvelope),
        "ampeg_decay" => parse_float_in_range(val, 0.0..=100.0)
            .map(AmpegDecay)
            .map(AmpegEnvelope),
        "ampeg_sustain" => parse_float_in_range(val, 0.0..=100.0)
            .map(AmpegSustain)
            .map(AmpegEnvelope),
        "ampeg_release" => parse_float_in_range(val, 0.0..=100.0)
            .map(AmpegRelease)
            .map(AmpegEnvelope),
        "ampeg_vel2release" => parse_float_in_range(val, -100.0..=100.0)
            .map(AmpegVel2Release)
            .map(AmpegEnvelope),

        "sample" => Some(Sample(val.replace('\\', "/"))),

        _ => None,
    })
}

fn parse_sfz_group(group: Group) -> Result<SfzGroupType, SfzValidationError> {
    Ok(match group.name.text {
        "region" => SfzGroupType::Region,
        "group" => SfzGroupType::Group,
        "master" => SfzGroupType::Master,
        "global" => SfzGroupType::Global,
        "control" => SfzGroupType::Control,
        "curve" => SfzGroupType::Curve,
        _ => SfzGroupType::Other,
    })
}

fn grammar_token_into_sfz_token(
    token: Token,
    defines: &HashMap<String, String>,
) -> Result<Option<SfzTokenWithMeta>, SfzValidationError> {
    match token.kind {
        TokenKind::Comment(_) => Ok(None),
        TokenKind::Group(group_type) => {
            Ok(Some(SfzTokenWithMeta::Group(parse_sfz_group(group_type)?)))
        }
        TokenKind::Opcode(opcode) => {
            Ok(parse_sfz_opcode(opcode, defines)?.map(SfzTokenWithMeta::Opcode))
        }
        TokenKind::Include(include) => Ok(Some(SfzTokenWithMeta::Import(
            include.path.text.replace('\\', "/"),
        ))),
        TokenKind::Define(define) => {
            let variable = define.variable.text.to_owned();
            let value = define.value.first.value.text.text.to_owned();
            //defines.borrow_mut().insert(variable.clone(), value.clone());
            Ok(Some(SfzTokenWithMeta::Define(variable, value)))
        }
    }
}

#[cfg(test)]
pub fn parse_tokens_raw<'a>(
    input: &'a str,
    defines: &'a HashMap<String, String>,
) -> impl 'a + Iterator<Item = Result<SfzTokenWithMeta, SfzParseError>> {
    let iter = ErrorTolerantToken::parse_as_iter(input);

    iter.filter_map(move |t| match t {
        Ok(t) => match grammar_token_into_sfz_token(t, defines) {
            Ok(Some(t)) => Some(Ok(t)),
            Ok(None) => None,
            Err(e) => Some(Err(SfzParseError::from(e))),
        },
        Err(e) => Some(Err(SfzParseError::from(e))),
    })
}

pub fn parse_tokens_resolved(file_path: &Path) -> Result<Vec<SfzToken>, SfzParseError> {
    TokenResolver::default().resolve_file(file_path)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{
        parse_key_number, parse_tokens_raw, parse_tokens_resolved, SfzToken, SfzTokenWithMeta,
    };

    fn create_temp_dir(label: &str) -> PathBuf {
        let unique = format!(
            "xsynth-sfz-parse-test-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn parse_key_number_accepts_note_names() {
        assert_eq!(parse_key_number("c4"), Some(60));
        assert_eq!(parse_key_number("db3"), Some(49));
        assert_eq!(parse_key_number("-1"), Some(-1));
    }

    #[test]
    fn parse_tokens_raw_resolves_defines_in_opcodes() {
        let defines = HashMap::from([("$VALUE".to_owned(), "snare.wav".to_owned())]);
        let input = "<region>\nsample=$VALUE\n";

        let tokens = parse_tokens_raw(input, &defines)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(matches!(tokens[0], SfzTokenWithMeta::Group(_)));
        assert!(matches!(
            tokens[1],
            SfzTokenWithMeta::Opcode(super::SfzOpcode::Sample(ref path)) if path == "snare.wav"
        ));
    }

    #[test]
    fn parse_tokens_resolved_includes_nested_files() {
        let dir = create_temp_dir("resolved");
        let root = dir.join("root.sfz");
        let included = dir.join("nested.sfz");

        fs::write(&included, "<region>\nsample=test.wav\n").unwrap();
        fs::write(&root, "#include \"nested.sfz\"\n").unwrap();

        let tokens = parse_tokens_resolved(&root).unwrap();

        assert!(matches!(
            tokens.as_slice(),
            [SfzToken::Group(_), SfzToken::Opcode(_)]
        ));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn parse_tokens_resolved_reparses_include_after_define_changes() {
        let dir = create_temp_dir("define-reparse");
        let root = dir.join("root.sfz");
        let included = dir.join("nested.sfz");

        fs::write(&included, "<region>\nsample=$NAME\n").unwrap();
        fs::write(
            &root,
            r#"
#define $NAME first.wav
#include "nested.sfz"
#define $NAME second.wav
#include "nested.sfz"
"#,
        )
        .unwrap();

        let tokens = parse_tokens_resolved(&root).unwrap();

        assert!(matches!(
            tokens[1],
            SfzToken::Opcode(super::SfzOpcode::Sample(ref path)) if path == "first.wav"
        ));
        assert!(matches!(
            tokens[3],
            SfzToken::Opcode(super::SfzOpcode::Sample(ref path)) if path == "second.wav"
        ));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn parse_tokens_resolved_rejects_include_cycles() {
        let dir = create_temp_dir("cycle");
        let root = dir.join("root.sfz");
        let nested = dir.join("nested.sfz");

        fs::write(&root, "#include \"nested.sfz\"\n").unwrap();
        fs::write(&nested, "#include \"root.sfz\"\n").unwrap();

        let result = parse_tokens_resolved(&root);
        let root_canonical = root.canonicalize().unwrap();

        assert!(matches!(
            result,
            Err(super::SfzParseError::IncludeCycle(path)) if path == root_canonical
        ));

        fs::remove_dir_all(dir).unwrap();
    }
}
