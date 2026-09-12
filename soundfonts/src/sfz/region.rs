use std::{
    collections::VecDeque,
    ops::RangeInclusive,
    path::{Path, PathBuf},
};

use std::collections::HashMap;
use std::sync::Arc;

use crate::{FilterType, LoopMode};

use super::parse::{AriaOnccBase, SfzAmpegEnvelope, SfzGroupType, SfzOpcode, SfzToken, XfCurve};

/// MOVE FORK: SFZ `trigger=` opcode. Regions default to Attack (spawn
/// on NoteOn). Release-trigger regions spawn on NoteOff instead and
/// generally play key-up samples (mechanical thumps, dampener noise).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerType {
    Attack,
    Release,
}

impl Default for TriggerType {
    fn default() -> Self {
        TriggerType::Attack
    }
}

/// Structure that holds the opcode parameters of the SFZ's AmpEG envelope.
#[derive(Debug, Clone)]
pub struct AmpegEnvelopeParams {
    pub ampeg_start: f32,
    pub ampeg_delay: f32,
    pub ampeg_attack: f32,
    pub ampeg_hold: f32,
    pub ampeg_decay: f32,
    pub ampeg_sustain: f32,
    pub ampeg_release: f32,
    pub ampeg_vel2release: f32,
}

impl Default for AmpegEnvelopeParams {
    fn default() -> Self {
        AmpegEnvelopeParams {
            ampeg_start: 0.0,
            ampeg_delay: 0.0,
            ampeg_attack: 0.01,
            ampeg_hold: 0.0,
            ampeg_decay: 0.0,
            ampeg_sustain: 100.0,
            ampeg_release: 0.01,
            ampeg_vel2release: 0.0,
        }
    }
}

impl AmpegEnvelopeParams {
    fn update_from_flag(&mut self, flag: SfzAmpegEnvelope) {
        match flag {
            SfzAmpegEnvelope::AmpegStart(val) => self.ampeg_start = val,
            SfzAmpegEnvelope::AmpegDelay(val) => self.ampeg_delay = val,
            SfzAmpegEnvelope::AmpegAttack(val) => self.ampeg_attack = val,
            SfzAmpegEnvelope::AmpegHold(val) => self.ampeg_hold = val,
            SfzAmpegEnvelope::AmpegDecay(val) => self.ampeg_decay = val,
            SfzAmpegEnvelope::AmpegSustain(val) => self.ampeg_sustain = val,
            SfzAmpegEnvelope::AmpegRelease(val) => self.ampeg_release = val,
            SfzAmpegEnvelope::AmpegVel2Release(val) => self.ampeg_vel2release = val,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RegionParamsBuilder {
    lovel: u8,
    hivel: u8,
    lokey: i8,
    hikey: i8,
    pitch_keycenter: i8,
    volume: i16,
    pan: i8,
    sample: Option<String>,
    default_path: Option<String>,
    loop_mode: LoopMode,
    loop_start: u32,
    loop_end: u32,
    /// MOVE FORK / Phase 8: SFZ `loop_crossfade=` in seconds. Converted
    /// to frame count downstream.
    loop_crossfade: f32,
    /// MOVE FORK / Phase 11: SFZ amp LFO (sine tremolo).
    amp_lfo_freq: f32,
    amp_lfo_depth: f32,
    /// MOVE FORK / Phase 11: live CC mod of amp LFO. Each entry is
    /// (CC, delta) — at CC=127 the LFO freq/depth becomes
    /// base + delta. Per-CC last-wins like volume_oncc.
    amp_lfo_freq_oncc: Vec<(u8, f32)>,
    amp_lfo_depth_oncc: Vec<(u8, f32)>,
    /// MOVE FORK / Phase 11: filter LFO (sine on cutoff).
    /// depth is in cents (SFZ standard); LiveCutoffState applies
    /// `freq *= 2^(sin · depth / 1200)`.
    fil_lfo_freq: f32,
    fil_lfo_depth: f32,
    fil_lfo_freq_oncc: Vec<(u8, f32)>,
    fil_lfo_depth_oncc: Vec<(u8, f32)>,
    /// MOVE FORK / Phase 11: pan LFO. depth in percent (-100..100).
    /// SIMDVoicePan adds `sin(phase) * depth / 100` into pan ∈ [-1, 1].
    pan_lfo_freq: f32,
    pan_lfo_depth: f32,
    pan_lfo_freq_oncc: Vec<(u8, f32)>,
    pan_lfo_depth_oncc: Vec<(u8, f32)>,
    /// MOVE FORK / Phase 11: filter envelope (ADSR autowah). depth in
    /// cents added to filter cutoff at envelope peak.
    fileg_attack: f32,
    fileg_decay: f32,
    fileg_sustain: f32,
    fileg_release: f32,
    fileg_depth: f32,
    /// MOVE FORK / 2026-05-16: curve id (into <curve> table map) used
    /// to shape env_level → cents lookup. None = direct multiply.
    fileg_curve: Option<u8>,
    /// MOVE FORK / Phase 11: pitch LFO (vibrato). depth in cents.
    pitch_lfo_freq: f32,
    pitch_lfo_depth: f32,
    pitch_lfo_freq_oncc: Vec<(u8, f32)>,
    pitch_lfo_depth_oncc: Vec<(u8, f32)>,
    offset: u32,
    cutoff: Option<f32>,
    resonance: f32,
    amp_veltrack: f32,
    amp_keycenter: i8,
    amp_keytrack: f32,
    pan_veltrack: f32,
    pan_keycenter: i8,
    pan_keytrack: f32,
    fil_veltrack: i16,
    fil_keycenter: i8,
    fil_keytrack: i16,
    filter_type: FilterType,
    ampeg_envelope: AmpegEnvelopeParams,
    tune: i16,
    trigger: TriggerType,
    /// MOVE FORK / 2026-05-19: SFZ `polyphony=N` opcode. None = not
    /// declared by the author. Soundfont collects the minimum value
    /// across all regions and surfaces it so the plugin can honor a
    /// monophonic SFZ (e.g. polyphony=1 at <global>).
    polyphony: Option<u32>,
    /// MOVE FORK / 2026-05-19: SFZ choke-group membership. None when
    /// the author didn't declare `group=`. See build() for how this
    /// collapses to an exclusive_class u8.
    choke_group: Option<u32>,
    /// MOVE FORK / 2026-05-19: SFZ `off_by=` — choking target group.
    choke_off_by: Option<u32>,
    seq_length: u32,
    seq_position: u32,
    /// MOVE FORK: per-CC range gates. `(lo, hi)` per CC number; region
    /// only fires if the static ARIA CC value (set_cc/set_hdcc at load)
    /// is in [lo, hi]. Evaluated at build(); regions whose constraints
    /// aren't satisfied are dropped entirely so they don't play
    /// unconditionally. Default empty = unconstrained.
    cc_ranges: HashMap<u8, (u8, u8)>,
    /// MOVE FORK: live `volume_oncc<N>=<dB>` bindings collected on this
    /// region. Each entry is (CC number, dB delta-per-unit). A future
    /// voice-side generator multiplies the voice amp by
    /// `db_to_amp(Σ delta·cc/127)`. Currently UNUSED at the call site —
    /// xsynth-core's voice spawners don't read this yet. Field exists so
    /// the parser/region/builder data path is in place before runtime
    /// support lands.
    volume_oncc: Vec<(u8, f32)>,
    /// MOVE FORK: live `cutoff_oncc<N>=<cents>` bindings on this region.
    /// See VolumeOncc for the accumulate / last-wins pattern.
    cutoff_oncc: Vec<(u8, f32)>,
    /// MOVE FORK: live `resonance_oncc<N>=<dB>` bindings on this region.
    resonance_oncc: Vec<(u8, f32)>,
    /// MOVE FORK: live `pan_oncc<N>=<percent>` bindings on this region.
    pan_oncc: Vec<(u8, f32)>,
    /// MOVE FORK / 2026-05-19: live `ampeg_*_oncc<N>=<value>` bindings.
    /// Each entry is (cc, value_per_unit, cc_init_at_parse_normalized).
    /// The runtime voice computes the *additional* contribution against
    /// the parse-time fold:
    ///     extra = value * (cc_now/127 - cc_init_at_parse)
    /// then adds to the base ampeg_* params. cc_init is captured at parse
    /// time so existing SGP-style libraries (set_hdcc72 + ampeg_release_oncc72)
    /// produce zero extra at the initial CC value (the fold already included
    /// that contribution) and respond linearly to subsequent CC changes.
    ampeg_attack_oncc:  Vec<(u8, f32, f32)>,
    ampeg_decay_oncc:   Vec<(u8, f32, f32)>,
    ampeg_sustain_oncc: Vec<(u8, f32, f32)>,
    ampeg_release_oncc: Vec<(u8, f32, f32)>,
    /// MOVE FORK / 2026-05-19: live `tune_oncc<N>=<cents>` bindings —
    /// each entry is (cc, cents_per_unit). Voice samples per-block (or
    /// per-spawn) and adds Σ delta·cc/127 cents to the base pitch
    /// ratio. Drives DS PITCH / GROUP_TUNING knob bindings.
    tune_oncc: Vec<(u8, f32)>,
    /// MOVE FORK: Phase 6 `_curvecc<N>=<curve_id>` bindings. Each entry
    /// (CC number, curve id) shapes the matching `_oncc<N>` modulation
    /// via curve[cc_value] ∈ [0, 1] instead of raw cc/127.
    volume_curvecc: Vec<(u8, u8)>,
    cutoff_curvecc: Vec<(u8, u8)>,
    resonance_curvecc: Vec<(u8, u8)>,
    pan_curvecc: Vec<(u8, u8)>,
    /// MOVE FORK / 2026-05-16: SFZ keyswitch opcodes. See SfzOpcode docs.
    /// `sw_last` is the constraint (None = unconstrained). `sw_default`
    /// is the initial keyswitch state. These inherit through the
    /// control→global→master→group→region stack like other opcodes.
    sw_last: Option<i8>,
    sw_default: Option<i8>,
    sw_lokey: Option<i8>,
    sw_hikey: Option<i8>,
    /// MOVE FORK / 2026-05-19: per-CC xfin/xfout crossfade ranges. Each
    /// entry is (lo, hi) — xsynth has no live xfin/xfout sweep so these
    /// resolve to a static amplitude factor at build time against the
    /// parser's `set_cc` state, folded into `volume`. Pianobook Fake
    /// Dulcimer (Tremolo) uses CC1 to pick which velocity-layer group
    /// fires: layers outside the current CC value get factor=0 and are
    /// dropped at build.
    xfin_cc:  HashMap<u8, (u8, u8)>,
    xfout_cc: HashMap<u8, (u8, u8)>,
    xf_cccurve: XfCurve,
}

impl Default for RegionParamsBuilder {
    fn default() -> Self {
        RegionParamsBuilder {
            lovel: 0,
            hivel: 127,
            lokey: 0,
            hikey: 127,
            pitch_keycenter: 60,
            volume: 0,
            pan: 0,
            sample: None,
            default_path: None,
            loop_mode: LoopMode::NoLoop,
            loop_start: 0,
            loop_end: 0,
            loop_crossfade: 0.0,
            amp_lfo_freq: 0.0,
            amp_lfo_depth: 0.0,
            amp_lfo_freq_oncc: Vec::new(),
            amp_lfo_depth_oncc: Vec::new(),
            fil_lfo_freq: 0.0,
            fil_lfo_depth: 0.0,
            fil_lfo_freq_oncc: Vec::new(),
            fil_lfo_depth_oncc: Vec::new(),
            pan_lfo_freq: 0.0,
            pan_lfo_depth: 0.0,
            pan_lfo_freq_oncc: Vec::new(),
            pan_lfo_depth_oncc: Vec::new(),
            fileg_attack: 0.0,
            fileg_decay: 0.0,
            fileg_sustain: 1.0,
            fileg_release: 0.0,
            fileg_depth: 0.0,
            fileg_curve: None,
            pitch_lfo_freq: 0.0,
            pitch_lfo_depth: 0.0,
            pitch_lfo_freq_oncc: Vec::new(),
            pitch_lfo_depth_oncc: Vec::new(),
            offset: 0,
            cutoff: None,
            resonance: 0.0,
            amp_veltrack: 100.0,
            amp_keycenter: 60,
            amp_keytrack: 0.0,
            pan_veltrack: 0.0,
            pan_keycenter: 60,
            pan_keytrack: 0.0,
            fil_veltrack: 0,
            fil_keycenter: 60,
            fil_keytrack: 0,
            filter_type: FilterType::default(),
            ampeg_envelope: AmpegEnvelopeParams::default(),
            tune: 0,
            trigger: TriggerType::Attack,
            polyphony: None,
            choke_group: None,
            choke_off_by: None,
            seq_length: 0,
            seq_position: 0,
            cc_ranges: HashMap::new(),
            volume_oncc: Vec::new(),
            cutoff_oncc: Vec::new(),
            ampeg_attack_oncc:  Vec::new(),
            ampeg_decay_oncc:   Vec::new(),
            ampeg_sustain_oncc: Vec::new(),
            ampeg_release_oncc: Vec::new(),
            tune_oncc: Vec::new(),
            resonance_oncc: Vec::new(),
            pan_oncc: Vec::new(),
            volume_curvecc: Vec::new(),
            cutoff_curvecc: Vec::new(),
            resonance_curvecc: Vec::new(),
            pan_curvecc: Vec::new(),
            sw_last: None,
            sw_default: None,
            sw_lokey: None,
            sw_hikey: None,
            xfin_cc:  HashMap::new(),
            xfout_cc: HashMap::new(),
            xf_cccurve: XfCurve::Gain,
        }
    }
}

impl RegionParamsBuilder {
    fn update_from_flag(&mut self, flag: SfzOpcode) {
        match flag {
            SfzOpcode::Lovel(val) => self.lovel = val,
            SfzOpcode::Hivel(val) => self.hivel = val,
            SfzOpcode::Key(val) => {
                self.lokey = val;
                self.hikey = val;
                self.pitch_keycenter = val;
            }
            SfzOpcode::Lokey(val) => self.lokey = val,
            SfzOpcode::Hikey(val) => self.hikey = val,
            SfzOpcode::PitchKeycenter(val) => self.pitch_keycenter = val,
            SfzOpcode::Pan(val) => self.pan = val,
            SfzOpcode::Volume(val) => self.volume = val,
            SfzOpcode::Sample(val) => self.sample = Some(val),
            SfzOpcode::LoopMode(val) => self.loop_mode = val,
            SfzOpcode::LoopStart(val) => self.loop_start = val,
            SfzOpcode::LoopEnd(val) => self.loop_end = val,
            SfzOpcode::LoopCrossfade(val) => self.loop_crossfade = val,
            SfzOpcode::AmpLfoFreq(val) => self.amp_lfo_freq = val,
            SfzOpcode::AmpLfoDepth(val) => self.amp_lfo_depth = val,
            SfzOpcode::FilLfoFreq(val) => self.fil_lfo_freq = val,
            SfzOpcode::FilLfoDepth(val) => self.fil_lfo_depth = val,
            SfzOpcode::FilLfoFreqOncc(cc, v) => {
                if let Some(existing) = self.fil_lfo_freq_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = v;
                } else {
                    self.fil_lfo_freq_oncc.push((cc, v));
                }
            }
            SfzOpcode::FilLfoDepthOncc(cc, v) => {
                if let Some(existing) = self.fil_lfo_depth_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = v;
                } else {
                    self.fil_lfo_depth_oncc.push((cc, v));
                }
            }
            SfzOpcode::AmpLfoFreqOncc(cc, v) => {
                if let Some(existing) = self.amp_lfo_freq_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = v;
                } else {
                    self.amp_lfo_freq_oncc.push((cc, v));
                }
            }
            SfzOpcode::AmpLfoDepthOncc(cc, v) => {
                if let Some(existing) = self.amp_lfo_depth_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = v;
                } else {
                    self.amp_lfo_depth_oncc.push((cc, v));
                }
            }
            SfzOpcode::PanLfoFreq(val) => self.pan_lfo_freq = val,
            SfzOpcode::PanLfoDepth(val) => self.pan_lfo_depth = val,
            SfzOpcode::PanLfoFreqOncc(cc, v) => {
                if let Some(existing) = self.pan_lfo_freq_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = v;
                } else {
                    self.pan_lfo_freq_oncc.push((cc, v));
                }
            }
            SfzOpcode::PanLfoDepthOncc(cc, v) => {
                if let Some(existing) = self.pan_lfo_depth_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = v;
                } else {
                    self.pan_lfo_depth_oncc.push((cc, v));
                }
            }
            SfzOpcode::FilegAttack(v)  => self.fileg_attack  = v,
            SfzOpcode::FilegDecay(v)   => self.fileg_decay   = v,
            SfzOpcode::FilegSustain(v) => self.fileg_sustain = v,
            SfzOpcode::FilegRelease(v) => self.fileg_release = v,
            SfzOpcode::FilegDepth(v)   => self.fileg_depth   = v,
            SfzOpcode::FilegCurve(id)  => self.fileg_curve   = Some(id),
            SfzOpcode::PitchLfoFreq(v)  => self.pitch_lfo_freq  = v,
            SfzOpcode::PitchLfoDepth(v) => self.pitch_lfo_depth = v,
            SfzOpcode::PitchLfoFreqOncc(cc, v) => {
                if let Some(existing) = self.pitch_lfo_freq_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = v;
                } else {
                    self.pitch_lfo_freq_oncc.push((cc, v));
                }
            }
            SfzOpcode::PitchLfoDepthOncc(cc, v) => {
                if let Some(existing) = self.pitch_lfo_depth_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = v;
                } else {
                    self.pitch_lfo_depth_oncc.push((cc, v));
                }
            }
            SfzOpcode::Offset(val) => self.offset = val,
            SfzOpcode::Cutoff(val) => self.cutoff = Some(val),
            SfzOpcode::Resonance(val) => self.resonance = val,
            SfzOpcode::AmpVeltrack(val) => self.amp_veltrack = val,
            SfzOpcode::AmpKeytrack(val) => self.amp_keytrack = val,
            SfzOpcode::AmpKeycenter(val) => self.amp_keycenter = val,
            SfzOpcode::PanVeltrack(val) => self.pan_veltrack = val,
            SfzOpcode::PanKeytrack(val) => self.pan_keytrack = val,
            SfzOpcode::PanKeycenter(val) => self.pan_keycenter = val,
            SfzOpcode::FilVeltrack(val) => self.fil_veltrack = val,
            SfzOpcode::FilKeytrack(val) => self.fil_keytrack = val,
            SfzOpcode::FilKeycenter(val) => self.fil_keycenter = val,
            SfzOpcode::FilterType(val) => self.filter_type = val,
            SfzOpcode::DefaultPath(val) => self.default_path = Some(val),
            SfzOpcode::AmpegEnvelope(flag) => self.ampeg_envelope.update_from_flag(flag),
            SfzOpcode::Tune(val) => self.tune = val,
            SfzOpcode::Polyphony(val) => self.polyphony = Some(val),
            // MOVE FORK / 2026-05-19: choke-group plumbing.
            // SFZ semantics:
            //   group=N    — region is a member of group N (0 = none)
            //   off_by=N   — note-on on this region chokes group N voices
            // We map both to a single u8 "exclusive_class" at build()
            // time using off_by if set (so the spawning region kills
            // the target group), else group (so the region's voice is
            // a target of others' off_by). The typical symmetric pattern
            // (group=1 off_by=1 on two regions) reduces to "same class
            // → mutually choke", matching the SF2 chokeGroup mechanism
            // xsynth already implements.
            SfzOpcode::SfzGroup(val) => {
                if val > 0 { self.choke_group = Some(val); }
            }
            SfzOpcode::SfzOffBy(val) => {
                if val > 0 { self.choke_off_by = Some(val); }
            }
            SfzOpcode::Trigger(val) => self.trigger = val,
            SfzOpcode::SeqLength(val) => self.seq_length = val,
            SfzOpcode::SeqPosition(val) => self.seq_position = val,
            SfzOpcode::LoCc(n, v) => {
                let entry = self.cc_ranges.entry(n).or_insert((0, 127));
                entry.0 = v;
            }
            SfzOpcode::HiCc(n, v) => {
                let entry = self.cc_ranges.entry(n).or_insert((0, 127));
                entry.1 = v;
            }
            // MOVE FORK: collect live volume_oncc bindings on this region.
            // Duplicate CCs replace the prior delta (SFZ "last assignment
            // wins"). Different CCs accumulate so a single region can be
            // modulated by multiple knobs at once.
            SfzOpcode::VolumeOncc(cc, db) => {
                if let Some(existing) = self.volume_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = db;
                } else {
                    self.volume_oncc.push((cc, db));
                }
            }
            SfzOpcode::CutoffOncc(cc, cents) => {
                if let Some(existing) = self.cutoff_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = cents;
                } else {
                    self.cutoff_oncc.push((cc, cents));
                }
            }
            SfzOpcode::ResonanceOncc(cc, db) => {
                if let Some(existing) = self.resonance_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = db;
                } else {
                    self.resonance_oncc.push((cc, db));
                }
            }
            SfzOpcode::PanOncc(cc, pct) => {
                if let Some(existing) = self.pan_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = pct;
                } else {
                    self.pan_oncc.push((cc, pct));
                }
            }
            // MOVE FORK / 2026-05-19: collect tune_oncc bindings on this region.
            SfzOpcode::TuneOncc(cc, cents) => {
                if let Some(existing) = self.tune_oncc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = cents;
                } else {
                    self.tune_oncc.push((cc, cents));
                }
            }
            // MOVE FORK: Phase 6 curvecc — attach curve_id to the binding
            // for the matching CC. Last-assignment-wins like _oncc.
            SfzOpcode::VolumeCurvecc(cc, id) => {
                if let Some(existing) = self.volume_curvecc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = id;
                } else {
                    self.volume_curvecc.push((cc, id));
                }
            }
            SfzOpcode::CutoffCurvecc(cc, id) => {
                if let Some(existing) = self.cutoff_curvecc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = id;
                } else {
                    self.cutoff_curvecc.push((cc, id));
                }
            }
            SfzOpcode::ResonanceCurvecc(cc, id) => {
                if let Some(existing) = self.resonance_curvecc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = id;
                } else {
                    self.resonance_curvecc.push((cc, id));
                }
            }
            SfzOpcode::PanCurvecc(cc, id) => {
                if let Some(existing) = self.pan_curvecc.iter_mut().find(|(c, _)| *c == cc) {
                    existing.1 = id;
                } else {
                    self.pan_curvecc.push((cc, id));
                }
            }
            // MOVE FORK: ARIA opcodes are resolved in parse_sf_root, not
            // here. If one reaches update_from_flag it means the parent
            // didn't intercept it — treat as no-op so the match stays
            // exhaustive.
            SfzOpcode::AriaCcInit(_, _) | SfzOpcode::AriaOncc { .. } => {}
            // <curve> block opcodes are routed into a CurveBuilder by
            // parse_sf_root, not into RegionParamsBuilder. Reach here
            // → silently no-op.
            SfzOpcode::CurveIndex(_) | SfzOpcode::CurvePoint(_, _) => {}
            // MOVE FORK / 2026-05-16: keyswitch opcodes. Inherit through
            // the SFZ hierarchy like other region fields; the v1 build
            // pass filters regions whose sw_last doesn't match the
            // global sw_default.
            SfzOpcode::SwLast(val)    => self.sw_last    = Some(val),
            SfzOpcode::SwDefault(val) => self.sw_default = Some(val),
            SfzOpcode::SwLokey(val)   => self.sw_lokey   = Some(val),
            SfzOpcode::SwHikey(val)   => self.sw_hikey   = Some(val),
            // MOVE FORK / 2026-05-19: CC-based crossfade ranges. Same
            // accumulation pattern as cc_ranges (LoCc/HiCc): each opcode
            // sets one side of the (lo, hi) pair for the given CC.
            SfzOpcode::XfInLoCc(n, v) => {
                let e = self.xfin_cc.entry(n).or_insert((0, 0));
                e.0 = v;
            }
            SfzOpcode::XfInHiCc(n, v) => {
                let e = self.xfin_cc.entry(n).or_insert((0, 0));
                e.1 = v;
            }
            SfzOpcode::XfOutLoCc(n, v) => {
                let e = self.xfout_cc.entry(n).or_insert((0, 0));
                e.0 = v;
            }
            SfzOpcode::XfOutHiCc(n, v) => {
                let e = self.xfout_cc.entry(n).or_insert((0, 0));
                e.1 = v;
            }
            SfzOpcode::XfCcCurve(c) => self.xf_cccurve = c,
        }
    }

    fn build(
        self,
        base_path: &Path,
        cc_state: &HashMap<u8, f32>,
        curves: Arc<HashMap<u8, [f32; 128]>>,
    ) -> Option<RegionParams> {
        // MOVE FORK: evaluate locc/hicc against static CC state. Regions
        // whose CC constraints aren't satisfied at default load-time CC
        // values get dropped (returning None). The CC value comes from
        // `set_cc<N>` / `set_hdcc<N>` opcodes; default 0 for CCs the
        // file didn't initialize. Splendid Grand Piano's resonance
        // group requires `locc64=65` (sustain pedal) and `locc70=1`
        // (resonance enable). Without this gating those regions fired
        // unconditionally and bled a doubled, octave-shifted sample
        // into every note.
        for (&cc_n, &(lo, hi)) in self.cc_ranges.iter() {
            let cc_v = cc_state.get(&cc_n).copied().unwrap_or(0.0);
            let cc_int = (cc_v * 127.0).round().clamp(0.0, 127.0) as u8;
            if cc_int < lo || cc_int > hi {
                return None;
            }
        }

        // MOVE FORK / 2026-05-19: static evaluation of xfin/xfout CC
        // crossfades. xsynth has no live xfin/xfout sweep, so each
        // configured (lo, hi) pair collapses to one amplitude factor
        // against the parser's set_cc state. Combined factor across all
        // CCs is folded into the region's volume as dB attenuation;
        // regions whose factor is effectively zero are dropped so we
        // don't pay sample I/O for silent layers. Drives the
        // velocity-layer pattern in Pianobook's Fake Dulcimer (Tremolo)
        // SFZ where set_cc1=5 picks one of four velocity-layer groups.
        let curve = self.xf_cccurve;
        let mut volume_factor: f32 = 1.0;
        for (&cc_n, &(lo, hi)) in self.xfin_cc.iter() {
            let cc_v = cc_state.get(&cc_n).copied().unwrap_or(0.0);
            let cc_int = (cc_v * 127.0).round().clamp(0.0, 127.0) as u8;
            volume_factor *= xfin_factor(cc_int, lo, hi, curve);
        }
        for (&cc_n, &(lo, hi)) in self.xfout_cc.iter() {
            let cc_v = cc_state.get(&cc_n).copied().unwrap_or(0.0);
            let cc_int = (cc_v * 127.0).round().clamp(0.0, 127.0) as u8;
            volume_factor *= xfout_factor(cc_int, lo, hi, curve);
        }
        // -80 dB threshold: any quieter and the voice is effectively
        // inaudible — drop the region entirely to save sample I/O.
        if volume_factor <= 1.0e-4 {
            return None;
        }
        // Fold into volume only when there's meaningful attenuation
        // (avoid integer-round noise when factor ≈ 1.0).
        let folded_volume = if (1.0 - volume_factor).abs() > 1.0e-3 {
            let db_attn = 20.0 * volume_factor.log10();
            ((self.volume as f32 + db_attn).round().clamp(-144.0, 6.0)) as i16
        } else {
            self.volume
        };

        let relative_sample_path = if let Some(default_path) = self.default_path {
            PathBuf::from(default_path).join(self.sample?)
        } else {
            self.sample?.into()
        };

        let mut sample_path = base_path.join(relative_sample_path);
        match sample_path.canonicalize() {
            Ok(path) => sample_path = path,
            Err(_) => return None,
        }

        Some(RegionParams {
            velrange: self.lovel..=self.hivel,
            keyrange: self.lokey..=self.hikey,
            pitch_keycenter: self.pitch_keycenter,
            volume: folded_volume,
            pan: self.pan,
            sample_path,
            loop_mode: self.loop_mode,
            loop_start: self.loop_start,
            loop_end: self.loop_end,
            loop_crossfade: self.loop_crossfade,
            amp_lfo_freq: self.amp_lfo_freq,
            amp_lfo_depth: self.amp_lfo_depth,
            amp_lfo_freq_oncc: self.amp_lfo_freq_oncc,
            amp_lfo_depth_oncc: self.amp_lfo_depth_oncc,
            fil_lfo_freq: self.fil_lfo_freq,
            fil_lfo_depth: self.fil_lfo_depth,
            fil_lfo_freq_oncc: self.fil_lfo_freq_oncc,
            fil_lfo_depth_oncc: self.fil_lfo_depth_oncc,
            pan_lfo_freq: self.pan_lfo_freq,
            pan_lfo_depth: self.pan_lfo_depth,
            pan_lfo_freq_oncc: self.pan_lfo_freq_oncc,
            pan_lfo_depth_oncc: self.pan_lfo_depth_oncc,
            fileg_attack: self.fileg_attack,
            fileg_decay: self.fileg_decay,
            fileg_sustain: self.fileg_sustain,
            fileg_release: self.fileg_release,
            fileg_depth: self.fileg_depth,
            fileg_curve: self.fileg_curve,
            pitch_lfo_freq: self.pitch_lfo_freq,
            pitch_lfo_depth: self.pitch_lfo_depth,
            pitch_lfo_freq_oncc: self.pitch_lfo_freq_oncc,
            pitch_lfo_depth_oncc: self.pitch_lfo_depth_oncc,
            offset: self.offset,
            cutoff: self.cutoff,
            resonance: self.resonance,
            amp_veltrack: self.amp_veltrack,
            amp_keycenter: self.amp_keycenter,
            amp_keytrack: self.amp_keytrack,
            pan_veltrack: self.pan_veltrack,
            pan_keycenter: self.pan_keycenter,
            pan_keytrack: self.pan_keytrack,
            fil_veltrack: self.fil_veltrack.clamp(-9600, 9600),
            fil_keycenter: self.fil_keycenter,
            fil_keytrack: self.fil_keytrack.clamp(0, 1200),
            filter_type: self.filter_type,
            ampeg_envelope: self.ampeg_envelope,
            tune: self.tune,
            trigger: self.trigger,
            polyphony: self.polyphony,
            // MOVE FORK / 2026-05-19: only set exclusive_class for the
            // SYMMETRIC choke pattern (group=N off_by=N on the same
            // region — typical closed/open hi-hat). xsynth's
            // exclusive_class mechanism conflates "I kill class X" with
            // "I'm in class X" into a single u8, so the asymmetric SFZ
            // case (off_by without matching group, or off_by != group)
            // collapses to broken self-choking: every region of the
            // off_by'd group ends up mutually killing each other.
            // For asymmetric chokes we'd need a real two-field model
            // (TODO); for now we drop the choke quietly rather than
            // breaking polyphony. RJS Classic Electric's
            // `tags="sustain" silencedByTags="legato"` lands here —
            // sustain regions go choke-less (polyphonic) and the
            // legato choke is a no-op (legato samples themselves are
            // filtered out earlier via the trigger=legato gate).
            exclusive_class: match (self.choke_off_by, self.choke_group) {
                (Some(o), Some(g)) if o == g => Some((o & 0xff) as u8),
                _ => None,
            },
            seq_length: self.seq_length,
            seq_position: self.seq_position,
            volume_oncc: self.volume_oncc,
            cutoff_oncc: self.cutoff_oncc,
            resonance_oncc: self.resonance_oncc,
            pan_oncc: self.pan_oncc,
            ampeg_attack_oncc:  self.ampeg_attack_oncc,
            ampeg_decay_oncc:   self.ampeg_decay_oncc,
            ampeg_sustain_oncc: self.ampeg_sustain_oncc,
            ampeg_release_oncc: self.ampeg_release_oncc,
            tune_oncc: self.tune_oncc,
            volume_curvecc: self.volume_curvecc,
            cutoff_curvecc: self.cutoff_curvecc,
            resonance_curvecc: self.resonance_curvecc,
            pan_curvecc: self.pan_curvecc,
            curves,
            sw_last: self.sw_last,
            sw_default: self.sw_default,
            sw_lokey: self.sw_lokey,
            sw_hikey: self.sw_hikey,
        })
    }
}

/// Structure that holds the opcode parameters of the SFZ file.
#[derive(Debug, Clone)]
pub struct RegionParams {
    pub velrange: RangeInclusive<u8>,
    pub keyrange: RangeInclusive<i8>,
    pub pitch_keycenter: i8,
    pub volume: i16,
    pub pan: i8,
    pub sample_path: PathBuf,
    pub loop_mode: LoopMode,
    pub loop_start: u32,
    pub loop_end: u32,
    /// MOVE FORK / Phase 8: loop crossfade duration in seconds.
    pub loop_crossfade: f32,
    /// MOVE FORK / Phase 11: amp LFO (sine tremolo). Zero freq /
    /// zero depth = inactive.
    pub amp_lfo_freq: f32,
    pub amp_lfo_depth: f32,
    /// MOVE FORK / Phase 11: live CC modulation of amp LFO params.
    pub amp_lfo_freq_oncc: Vec<(u8, f32)>,
    pub amp_lfo_depth_oncc: Vec<(u8, f32)>,
    /// MOVE FORK / Phase 11: filter LFO (sine on cutoff, depth in cents).
    pub fil_lfo_freq: f32,
    pub fil_lfo_depth: f32,
    pub fil_lfo_freq_oncc: Vec<(u8, f32)>,
    pub fil_lfo_depth_oncc: Vec<(u8, f32)>,
    /// MOVE FORK / Phase 11: pan LFO (sine on pan, depth in percent).
    pub pan_lfo_freq: f32,
    pub pan_lfo_depth: f32,
    pub pan_lfo_freq_oncc: Vec<(u8, f32)>,
    pub pan_lfo_depth_oncc: Vec<(u8, f32)>,
    /// MOVE FORK / Phase 11: filter ADSR routed to cutoff cents
    /// (autowah). Depth = full-swing cents at peak.
    pub fileg_attack: f32,
    pub fileg_decay: f32,
    pub fileg_sustain: f32,
    pub fileg_release: f32,
    pub fileg_depth: f32,
    /// MOVE FORK / 2026-05-16: optional curve table index for the filter
    /// envelope. When set, LiveCutoffState uses `curves[id][env_level*127]`
    /// as a 0..1 multiplier on `fileg_depth` instead of feeding env_level
    /// directly into the exponential cents domain. Lets DS's
    /// `<envelope translation="linear">` (linear-Hz sweep) be pre-baked
    /// into the SFZ exp-cents pipeline.
    pub fileg_curve: Option<u8>,
    /// MOVE FORK / Phase 11: pitch LFO (vibrato) — sine multiplier
    /// into voice pitch_fac, depth in cents.
    pub pitch_lfo_freq: f32,
    pub pitch_lfo_depth: f32,
    pub pitch_lfo_freq_oncc: Vec<(u8, f32)>,
    pub pitch_lfo_depth_oncc: Vec<(u8, f32)>,
    pub offset: u32,
    pub cutoff: Option<f32>,
    pub resonance: f32,
    pub amp_veltrack: f32,
    pub amp_keycenter: i8,
    pub amp_keytrack: f32,
    pub pan_veltrack: f32,
    pub pan_keycenter: i8,
    pub pan_keytrack: f32,
    pub fil_veltrack: i16,
    pub fil_keycenter: i8,
    pub fil_keytrack: i16,
    pub filter_type: FilterType,
    pub ampeg_envelope: AmpegEnvelopeParams,
    pub tune: i16,
    pub trigger: TriggerType,
    /// MOVE FORK / 2026-05-19: SFZ `polyphony=N` opcode (None = author
    /// didn't declare). Soundfont aggregates min across all regions and
    /// the plugin uses it to override the auto-heuristic when present.
    pub polyphony: Option<u32>,
    /// MOVE FORK / 2026-05-19: SFZ `group=` + `off_by=` mapped down to
    /// a single u8 exclusive_class. xsynth-core's voice spawn checks
    /// `exclusive_class` and kills any prior voices that share it —
    /// covers the typical "open/closed hi-hat" SFZ choke pattern
    /// (both regions group=1 off_by=1) directly. None = no choke.
    pub exclusive_class: Option<u8>,
    /// MOVE FORK: SFZ round-robin opcodes. `seq_length` is the total
    /// number of RR variations (0 = no RR). `seq_position` is this
    /// region's 1-based slot in the sequence (0 = always-fire, ignored
    /// for RR rotation).
    pub seq_length: u32,
    pub seq_position: u32,
    /// MOVE FORK: live `volume_oncc<N>=<dB>` bindings. See
    /// RegionParamsBuilder.volume_oncc for semantics. Unread by
    /// xsynth-core today — will be consumed by a SIMD generator in a
    /// later Phase 3 step.
    pub volume_oncc: Vec<(u8, f32)>,
    /// MOVE FORK: live `cutoff_oncc<N>=<cents>` bindings on this region.
    pub cutoff_oncc: Vec<(u8, f32)>,
    /// MOVE FORK: live `resonance_oncc<N>=<dB>` bindings on this region.
    pub resonance_oncc: Vec<(u8, f32)>,
    /// MOVE FORK: live `pan_oncc<N>=<percent>` bindings on this region.
    pub pan_oncc: Vec<(u8, f32)>,
    /// MOVE FORK / 2026-05-19: live `ampeg_*_oncc<N>=<value>` bindings.
    /// Each entry is (cc, value_per_unit, cc_init_at_parse_normalized).
    /// Voice computes the runtime contribution as
    ///     extra = value * (cc_now/127 - cc_init_at_parse)
    /// and adds it to the base ampeg_* param at note-on (attack/decay/
    /// sustain snapshot) and note-off (release recompute).
    pub ampeg_attack_oncc:  Vec<(u8, f32, f32)>,
    pub ampeg_decay_oncc:   Vec<(u8, f32, f32)>,
    pub ampeg_sustain_oncc: Vec<(u8, f32, f32)>,
    pub ampeg_release_oncc: Vec<(u8, f32, f32)>,
    /// MOVE FORK / 2026-05-19: live `tune_oncc<N>=<cents>` bindings.
    /// (cc, cents_per_unit). Voice adds Σ delta·cc/127 cents to base
    /// pitch — drives DS PITCH/GROUP_TUNING knob bindings.
    pub tune_oncc: Vec<(u8, f32)>,
    /// MOVE FORK: Phase 6 curvecc bindings — each entry (CC, curve_id)
    /// tells the matching SIMD generator to look up curve[cc_value]
    /// from `curves` instead of using cc_value/127 directly.
    pub volume_curvecc: Vec<(u8, u8)>,
    pub cutoff_curvecc: Vec<(u8, u8)>,
    pub resonance_curvecc: Vec<(u8, u8)>,
    pub pan_curvecc: Vec<(u8, u8)>,
    /// MOVE FORK: Phase 6 `<curve>` blocks parsed from the SFZ. Shared
    /// across all regions via Arc — region clones are cheap refcount
    /// bumps. Each curve is a 128-point lookup table.
    pub curves: Arc<HashMap<u8, [f32; 128]>>,
    /// MOVE FORK / 2026-05-16: keyswitch constraints. `sw_last = Some(K)`
    /// means this region only fires when the most-recent keyswitch press
    /// was key K. `sw_default = Some(K)` is the initial keyswitch state
    /// for the file (typically set in `<global>`). v1: xsynth-core drops
    /// regions whose sw_last != sw_default at preset build time.
    pub sw_last: Option<i8>,
    pub sw_default: Option<i8>,
    pub sw_lokey: Option<i8>,
    pub sw_hikey: Option<i8>,
}

/// MOVE FORK / 2026-05-19: xfin amplitude factor at CC value `cc` for a
/// (lo, hi) fade-in range. Below lo → 0 (silent), above hi → 1 (full),
/// linear in amplitude between (gain curve) or sin(pi/2 * frac) for
/// equal-power (power curve). Degenerate range (hi <= lo) collapses to a
/// binary step at lo.
fn xfin_factor(cc: u8, lo: u8, hi: u8, curve: XfCurve) -> f32 {
    if hi <= lo {
        // SFZ convention: an unconfigured range (still at (0, 0)) means
        // "fade-in inactive, full volume". A configured but inverted /
        // degenerate range falls back to a binary step at `lo`.
        if lo == 0 && hi == 0 { return 1.0; }
        return if cc >= lo { 1.0 } else { 0.0 };
    }
    if cc <= lo { return 0.0; }
    if cc >= hi { return 1.0; }
    let frac = (cc - lo) as f32 / (hi - lo) as f32;
    match curve {
        XfCurve::Gain  => frac,
        XfCurve::Power => (frac * std::f32::consts::FRAC_PI_2).sin(),
    }
}

/// MOVE FORK / 2026-05-19: xfout amplitude factor — mirror of xfin.
/// Below lo → 1 (still full), above hi → 0 (silent), interpolating
/// between. Degenerate range collapses to a binary step at lo.
fn xfout_factor(cc: u8, lo: u8, hi: u8, curve: XfCurve) -> f32 {
    if hi <= lo {
        if lo == 0 && hi == 0 { return 1.0; }
        return if cc > lo { 0.0 } else { 1.0 };
    }
    if cc <= lo { return 1.0; }
    if cc >= hi { return 0.0; }
    let frac = (hi - cc) as f32 / (hi - lo) as f32;
    match curve {
        XfCurve::Gain  => frac,
        XfCurve::Power => (frac * std::f32::consts::FRAC_PI_2).sin(),
    }
}

fn get_group_level(group_type: SfzGroupType) -> Option<usize> {
    match group_type {
        SfzGroupType::Control => Some(1),
        SfzGroupType::Global => Some(2),
        SfzGroupType::Master => Some(3),
        SfzGroupType::Group => Some(4),
        SfzGroupType::Region => Some(5),
        // MOVE FORK: <curve> is a sibling of <control>, parsed by
        // parse_sf_root via a separate code path. Returning None here
        // keeps it out of the region inheritance stack.
        SfzGroupType::Curve => None,
        SfzGroupType::Other => None,
    }
}

pub(super) fn parse_sf_root(
    tokens: impl Iterator<Item = SfzToken>,
    base_path: PathBuf,
) -> Vec<RegionParams> {
    let mut current_group = None;
    let mut group_data_stack = VecDeque::<RegionParamsBuilder>::new();
    let mut regions = Vec::new();
    // MOVE FORK: ARIA CC initial state, populated by `set_cc<N>` /
    // `set_hdcc<N>` from the <control> block, consumed by `_oncc`
    // modulators to bake their static contribution. Default 0 for any
    // CC the file didn't initialize.
    let mut cc_state: HashMap<u8, f32> = HashMap::new();
    // MOVE FORK / Phase 6: <curve> block state. The active curve is
    // identified by `index=` and populated via `vNNN=` opcodes. On exit
    // (any non-curve group token after curve content), the assembled
    // table moves into `curves`.
    let mut curves: HashMap<u8, [f32; 128]> = HashMap::new();
    let mut current_curve_id: Option<u8> = None;
    let mut current_curve_values: [f32; 128] = [0.0; 128];

    for token in tokens {
        match token {
            SfzToken::Group(group) => {
                // Finalize an in-progress curve when leaving the block.
                if current_group == Some(SfzGroupType::Curve) {
                    if let Some(id) = current_curve_id.take() {
                        curves.insert(id, current_curve_values);
                    }
                    current_curve_values = [0.0; 128];
                }
                if current_group == Some(SfzGroupType::Region) {
                    let next_region = group_data_stack.pop_back().unwrap();
                    if let Some(built) = next_region.build(
                        &base_path,
                        &cc_state,
                        Arc::new(HashMap::new()), // placeholder; filled at end below via re-walk
                    ) {
                        regions.push(built);
                    }
                }

                if group == SfzGroupType::Curve {
                    current_group = Some(SfzGroupType::Curve);
                } else if let Some(group_level) = get_group_level(group) {
                    current_group = Some(group);

                    // MOVE FORK: when entering a new header at level N
                    // (e.g. a fresh `<group>` after a previous `<group>`),
                    // pop the stale level-N (and any deeper) entry first
                    // so the new entry inherits from its parent (level
                    // N-1) rather than carrying the previous sibling's
                    // opcodes. Without this, `trigger=release` from one
                    // group leaked into every subsequent group, making
                    // full attack-trigger samples fire on note-off.
                    while group_data_stack.len() >= group_level {
                        group_data_stack.pop_back();
                    }
                    while group_data_stack.len() < group_level {
                        let parent_group = group_data_stack.back().cloned().unwrap_or_default();
                        group_data_stack.push_back(parent_group);
                    }
                } else {
                    current_group = None;
                }
            }
            // MOVE FORK / Phase 6: curve block content.
            SfzToken::Opcode(SfzOpcode::CurveIndex(id))
                if current_group == Some(SfzGroupType::Curve) =>
            {
                // Flush any prior unfinalized curve (same block, new index
                // — unusual but defensive).
                if let Some(prev) = current_curve_id.take() {
                    curves.insert(prev, current_curve_values);
                    current_curve_values = [0.0; 128];
                }
                current_curve_id = Some(id);
            }
            SfzToken::Opcode(SfzOpcode::CurvePoint(n, v))
                if current_group == Some(SfzGroupType::Curve) =>
            {
                if (n as usize) < 128 {
                    current_curve_values[n as usize] = v;
                }
            }
            SfzToken::Opcode(SfzOpcode::AriaCcInit(cc_n, value)) => {
                cc_state.insert(cc_n, value);
            }
            SfzToken::Opcode(SfzOpcode::AriaOncc { base, cc, value }) => {
                let cc_v = cc_state.get(&cc).copied().unwrap_or(0.0);
                let contribution = value * cc_v;
                if current_group.is_some() && current_group != Some(SfzGroupType::Curve) {
                    if let Some(group_data) = group_data_stack.back_mut() {
                        let env = &mut group_data.ampeg_envelope;
                        // (1) Fold into base ampeg_* so libraries that rely on
                        // a single static set_hdcc<N>=v (Splendid Grand Piano)
                        // still produce the right initial envelope. The runtime
                        // contribution below subtracts cc_init back out, so the
                        // base + extra at CC==init equals the folded value.
                        match base {
                            AriaOnccBase::AmpegAttack => env.ampeg_attack += contribution,
                            AriaOnccBase::AmpegHold => env.ampeg_hold += contribution,
                            AriaOnccBase::AmpegDecay => env.ampeg_decay += contribution,
                            AriaOnccBase::AmpegSustain => env.ampeg_sustain += contribution,
                            AriaOnccBase::AmpegRelease => env.ampeg_release += contribution,
                            AriaOnccBase::AmpegDelay => env.ampeg_delay += contribution,
                            AriaOnccBase::AmpegStart => env.ampeg_start += contribution,
                        }
                        // (2) MOVE FORK / 2026-05-19: also record the binding
                        // for the runtime voice. cc_v is the normalized init
                        // value used in the fold; runtime evaluates
                        //     extra = value * (cc_now/127 - cc_v)
                        // and adds to the base ampeg_* param at envelope
                        // stage transitions (note-on for atk/dec/sus, note-off
                        // for release). attack/decay/sustain/release only;
                        // hold/delay/start aren't exposed as live knobs.
                        //
                        // MOVE FORK / 2026-09-12: only for CC < 128. An
                        // ARIA extended CC (Splendid Grand Piano's
                        // `ampeg_decay_oncc133`) has no runtime half at
                        // all: `CcState` holds 128 atomics and no MIDI
                        // message can set CC >= 128, so its value is
                        // frozen at `set_hdcc<N>` forever and the fold
                        // above is already the exact answer. Recording
                        // the binding would index out of bounds at voice
                        // spawn — a panic on the audio thread at note-on.
                        let entry = (cc, value, cc_v);
                        fn upsert(v: &mut Vec<(u8, f32, f32)>, e: (u8, f32, f32)) {
                            if let Some(x) = v.iter_mut().find(|x| x.0 == e.0) {
                                *x = e;
                            } else {
                                v.push(e);
                            }
                        }
                        if cc < 128 {
                            match base {
                                AriaOnccBase::AmpegAttack  => upsert(&mut group_data.ampeg_attack_oncc,  entry),
                                AriaOnccBase::AmpegDecay   => upsert(&mut group_data.ampeg_decay_oncc,   entry),
                                AriaOnccBase::AmpegSustain => upsert(&mut group_data.ampeg_sustain_oncc, entry),
                                AriaOnccBase::AmpegRelease => upsert(&mut group_data.ampeg_release_oncc, entry),
                                _ => {}
                            }
                        }
                    }
                }
            }
            SfzToken::Opcode(flag) => {
                if current_group.is_some() && current_group != Some(SfzGroupType::Curve) {
                    if let Some(group_data) = group_data_stack.back_mut() {
                        group_data.update_from_flag(flag);
                    }
                }
            }
        }
    }

    // Finalize trailing curve if the file ends inside one.
    if current_group == Some(SfzGroupType::Curve) {
        if let Some(id) = current_curve_id.take() {
            curves.insert(id, current_curve_values);
        }
    }

    if current_group == Some(SfzGroupType::Region) {
        let next_region = group_data_stack.pop_back().unwrap();
        if let Some(built) = next_region.build(
            &base_path,
            &cc_state,
            Arc::new(HashMap::new()),
        ) {
            regions.push(built);
        }
    }

    // MOVE FORK / Phase 6: backfill the shared curves Arc into every
    // region. Curves can appear anywhere in the document (commonly
    // before <region>s but the spec doesn't require it), so we build
    // them up over the whole pass and attach at the end.
    let curves_arc = Arc::new(curves);
    for r in regions.iter_mut() {
        r.curves = curves_arc.clone();
    }

    regions
}
