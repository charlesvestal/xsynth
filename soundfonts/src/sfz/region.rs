use std::{
    collections::VecDeque,
    ops::RangeInclusive,
    path::{Path, PathBuf},
};

use std::collections::HashMap;
use std::sync::Arc;

use crate::{FilterType, LoopMode};

use super::parse::{AriaOnccBase, SfzAmpegEnvelope, SfzGroupType, SfzOpcode, SfzToken};

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
    /// MOVE FORK: Phase 6 `_curvecc<N>=<curve_id>` bindings. Each entry
    /// (CC number, curve id) shapes the matching `_oncc<N>` modulation
    /// via curve[cc_value] ∈ [0, 1] instead of raw cc/127.
    volume_curvecc: Vec<(u8, u8)>,
    cutoff_curvecc: Vec<(u8, u8)>,
    resonance_curvecc: Vec<(u8, u8)>,
    pan_curvecc: Vec<(u8, u8)>,
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
            seq_length: 0,
            seq_position: 0,
            cc_ranges: HashMap::new(),
            volume_oncc: Vec::new(),
            cutoff_oncc: Vec::new(),
            resonance_oncc: Vec::new(),
            pan_oncc: Vec::new(),
            volume_curvecc: Vec::new(),
            cutoff_curvecc: Vec::new(),
            resonance_curvecc: Vec::new(),
            pan_curvecc: Vec::new(),
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
            volume: self.volume,
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
            seq_length: self.seq_length,
            seq_position: self.seq_position,
            volume_oncc: self.volume_oncc,
            cutoff_oncc: self.cutoff_oncc,
            resonance_oncc: self.resonance_oncc,
            pan_oncc: self.pan_oncc,
            volume_curvecc: self.volume_curvecc,
            cutoff_curvecc: self.cutoff_curvecc,
            resonance_curvecc: self.resonance_curvecc,
            pan_curvecc: self.pan_curvecc,
            curves,
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
                        match base {
                            AriaOnccBase::AmpegAttack => env.ampeg_attack += contribution,
                            AriaOnccBase::AmpegHold => env.ampeg_hold += contribution,
                            AriaOnccBase::AmpegDecay => env.ampeg_decay += contribution,
                            AriaOnccBase::AmpegSustain => env.ampeg_sustain += contribution,
                            AriaOnccBase::AmpegRelease => env.ampeg_release += contribution,
                            AriaOnccBase::AmpegDelay => env.ampeg_delay += contribution,
                            AriaOnccBase::AmpegStart => env.ampeg_start += contribution,
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
