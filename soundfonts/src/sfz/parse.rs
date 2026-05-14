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
    Offset(u32),
    Cutoff(f32),
    Resonance(f32),
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
    /// MOVE FORK: `index=<id>` inside a `<curve>` block. Identifies the
    /// curve table being defined.
    CurveIndex(u8),
    /// MOVE FORK: `v<NNN>=<f32>` inside a `<curve>` block. NNN is
    /// 0..127. Defines one sample point of the current curve.
    CurvePoint(u8, f32),
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
            let base = match base_name {
                "ampeg_attack" => Some(AriaOnccBase::AmpegAttack),
                "ampeg_hold" => Some(AriaOnccBase::AmpegHold),
                "ampeg_decay" => Some(AriaOnccBase::AmpegDecay),
                "ampeg_sustain" => Some(AriaOnccBase::AmpegSustain),
                "ampeg_release" | "ampeg_releasecc" => Some(AriaOnccBase::AmpegRelease),
                "ampeg_delay" => Some(AriaOnccBase::AmpegDelay),
                "ampeg_start" => Some(AriaOnccBase::AmpegStart),
                _ => None,
            };
            if let Some(base) = base {
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
        "offset" => parse_u32_in_range(val, 0..=u32::MAX).map(Offset),
        "default_path" => Some(DefaultPath(val.replace('\\', "/"))),
        "tune" => parse_i16_in_range(val, -2400..=2400).map(Tune),
        "trigger" => parse_trigger(val).map(Trigger),
        "seq_length" | "seqlength" => parse_u32_in_range(val, 0..=255).map(SeqLength),
        "seq_position" | "seqposition" => parse_u32_in_range(val, 0..=255).map(SeqPosition),

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
