use std::{marker::PhantomData, sync::{atomic::Ordering, Arc}};

use simdeez::prelude::*;

use crate::{
    effects::BiQuadFilter,
    helpers::db_to_amp,
    voice::{CcState, ReleaseType, SIMDVoiceGenerator, VoiceControlData},
};

use super::{SIMDSampleMono, SIMDSampleStereo, VoiceGeneratorBase};

use xsynth_soundfonts::FilterType;
use biquad::Q_BUTTERWORTH_F32;

/// MOVE FORK: see oncc_amp.rs / pan_oncc.rs for cadence rationale.
const RECOMPUTE_INTERVAL: u32 = 8;

/// MOVE FORK / Phase 6: resolve CC value through optional curve.
#[inline]
fn cc_lookup(
    cc_val: u8,
    cc: u8,
    curvecc: &[(u8, u8)],
    curves: &std::collections::HashMap<u8, [f32; 128]>,
) -> f32 {
    if let Some((_, id)) = curvecc.iter().find(|(c, _)| *c == cc) {
        if let Some(table) = curves.get(id) {
            return table[cc_val as usize];
        }
    }
    cc_val as f32 / 127.0
}

/// MOVE FORK: live filter state shared between mono/stereo cutoff
/// modulators.
///
/// On each tick (every `RECOMPUTE_INTERVAL` SIMD-vector calls = ~32
/// audio samples at WIDTH=4):
///   1. Compute target freq + Q from current CC state.
///   2. One-pole-follow current freq + Q toward target with
///      `alpha = 1 - exp(-tick_samples / smooth_samples)` ≈ 0.07
///      (~10ms smoothing time constant at 44.1kHz).
///   3. Re-derive biquad coefficients from the smoothed values.
///
/// Without the follower the filter coefficients jump 32 samples at a
/// time, which is audibly zippery on knob sweeps. The exponential
/// follower spreads each step across ~10ms of smoothing.
struct LiveCutoffState {
    cc_state: CcState,
    cutoff_oncc: Arc<[(u8, f32)]>,
    resonance_oncc: Arc<[(u8, f32)]>,
    cutoff_curvecc: Arc<[(u8, u8)]>,
    resonance_curvecc: Arc<[(u8, u8)]>,
    curves: Arc<std::collections::HashMap<u8, [f32; 128]>>,
    /// MOVE FORK / Phase 11: filter LFO source — sine wave at
    /// `fil_lfo_freq` Hz, depth in cents. Applied multiplicatively
    /// to the cutoff: `freq *= 2^(sin · depth / 1200)`.
    fil_lfo_freq: f32,
    fil_lfo_depth: f32,
    fil_lfo_freq_oncc: Arc<[(u8, f32)]>,
    fil_lfo_depth_oncc: Arc<[(u8, f32)]>,
    /// Running phase in radians.
    lfo_phase: f32,
    /// MOVE FORK / Phase 11: filter envelope (autowah). Times in
    /// seconds, depth in cents added at peak; advanced one tick at a
    /// time inside step(). `fileg_released` flips on signal_release().
    fileg_attack: f32,
    fileg_decay: f32,
    fileg_sustain: f32,
    fileg_release: f32,
    fileg_depth: f32,
    /// MOVE FORK / 2026-05-16: optional curve lookup applied to the
    /// envelope level before multiplying by depth. None = use level
    /// directly (SFZ-spec exp-cents sweep). Some(id) = use
    /// `curves[id][round(level*127)]` so DS's `translation="linear"`
    /// (linear-Hz) envelopes can be pre-baked by the converter.
    fileg_curve: Option<u8>,
    fileg_stage: u8,    // 0=A 1=D 2=S 3=R 4=done
    fileg_level: f32,   // current 0..1
    fileg_released: bool,
    fil_type: FilterType,
    sample_rate: f32,
    base_freq: f32,
    base_resonance_db: f32,
    /// Smoothed current freq (Hz) — what the biquad coefficients
    /// reflect this tick.
    current_freq: f32,
    /// Smoothed current Q.
    current_q: f32,
    /// Pre-computed one-pole alpha per tick.
    smooth_alpha: f32,
    static_filter: bool,
    countdown: u32,
}

impl LiveCutoffState {
    fn new(
        cc_state: CcState,
        cutoff_oncc: Arc<[(u8, f32)]>,
        resonance_oncc: Arc<[(u8, f32)]>,
        cutoff_curvecc: Arc<[(u8, u8)]>,
        resonance_curvecc: Arc<[(u8, u8)]>,
        curves: Arc<std::collections::HashMap<u8, [f32; 128]>>,
        fil_lfo_freq: f32,
        fil_lfo_depth: f32,
        fil_lfo_freq_oncc: Arc<[(u8, f32)]>,
        fil_lfo_depth_oncc: Arc<[(u8, f32)]>,
        fileg_attack: f32,
        fileg_decay: f32,
        fileg_sustain: f32,
        fileg_release: f32,
        fileg_depth: f32,
        fileg_curve: Option<u8>,
        fil_type: FilterType,
        sample_rate: f32,
        base_freq: f32,
        base_resonance_db: f32,
    ) -> Self {
        let max_lfo_freq_delta: f32 = fil_lfo_freq_oncc.iter().map(|(_, d)| d.abs()).sum();
        let max_lfo_depth_delta: f32 = fil_lfo_depth_oncc.iter().map(|(_, d)| d.abs()).sum();
        let has_lfo = (fil_lfo_freq + max_lfo_freq_delta) > 0.0
            && (fil_lfo_depth + max_lfo_depth_delta) > 0.0;
        let has_fileg = fileg_depth.abs() > 0.0;
        let static_filter = cutoff_oncc.is_empty() && resonance_oncc.is_empty()
            && !has_lfo && !has_fileg;
        // 10ms smoothing time constant. tick_samples = RECOMPUTE_INTERVAL
        // * a SIMD vector width (assume 4 for aarch64 NEON; the exact
        // value isn't critical — alpha just controls smoothing speed,
        // any reasonable SIMD width gives a similar audible time
        // constant).
        let tick_samples = (RECOMPUTE_INTERVAL as f32) * 4.0;
        let smooth_samples = sample_rate * 0.010;
        let smooth_alpha = 1.0 - (-tick_samples / smooth_samples).exp();
        let init_q = db_to_amp(base_resonance_db) * Q_BUTTERWORTH_F32;
        let mut s = Self {
            cc_state,
            cutoff_oncc,
            resonance_oncc,
            cutoff_curvecc,
            resonance_curvecc,
            curves,
            fil_lfo_freq,
            fil_lfo_depth,
            fil_lfo_freq_oncc,
            fil_lfo_depth_oncc,
            lfo_phase: 0.0,
            fileg_attack,
            fileg_decay,
            fileg_sustain,
            fileg_release,
            fileg_depth,
            fileg_curve,
            fileg_stage: 0,
            fileg_level: 0.0,
            fileg_released: false,
            fil_type,
            sample_rate,
            base_freq,
            base_resonance_db,
            current_freq: base_freq,
            current_q: init_q,
            smooth_alpha,
            static_filter,
            countdown: 0,
        };
        // MOVE FORK / 2026-05-16: snap current freq/Q to the live CC
        // target on construction. Otherwise every voice spawn starts
        // the filter at the SFZ static cutoff= / resonance= base
        // (which our converter writes as knob_min — typically near
        // silence) and smooths it toward the CC-resolved target over
        // ~45 ms, producing an audible filter-envelope sweep on each
        // note-on that doesn't exist in DS. K4-Acoustic surfaced this:
        // the cutoff knob's `value=22000` only takes effect via CC,
        // so without this snap the filter starts at 1 Hz and rings
        // open during voice attack. Skip when static_filter (no oncc
        // bindings → nothing to chase) so existing static-filter
        // behavior is unchanged.
        if !s.static_filter {
            let (tf, tq) = s.target();
            s.current_freq = tf;
            s.current_q = tq;
            // target() advanced lfo_phase by one tick; rewind so the
            // first audio tick starts at phase 0 like before.
            s.lfo_phase = 0.0;
        }
        s
    }

    /// MOVE FORK / Phase 11: forward note-off into the filter envelope.
    fn notify_release(&mut self) {
        self.fileg_released = true;
    }

    /// Target (freq, Q) from the live CC state + filter LFO.
    /// Mutates `lfo_phase`; call once per tick.
    #[inline]
    fn target(&mut self) -> (f32, f32) {
        let mut cents_sum = 0.0_f32;
        for (cc, delta) in self.cutoff_oncc.iter() {
            let cc_val = self.cc_state[*cc as usize].load(Ordering::Relaxed);
            cents_sum += delta * cc_lookup(cc_val, *cc, &self.cutoff_curvecc, &self.curves);
        }
        // MOVE FORK / Phase 11: filter LFO contribution. Compute live
        // freq + depth from base + oncc, advance phase by the per-tick
        // amount (tick = RECOMPUTE_INTERVAL * SIMD vector = ~32 audio
        // frames at NEON WIDTH=4), then add sin(phase) · depth cents
        // to cents_sum.
        let mut lfo_freq = self.fil_lfo_freq;
        for (cc, delta) in self.fil_lfo_freq_oncc.iter() {
            let cc_val = self.cc_state[*cc as usize].load(Ordering::Relaxed) as f32 / 127.0;
            lfo_freq += delta * cc_val;
        }
        let mut lfo_depth = self.fil_lfo_depth;
        for (cc, delta) in self.fil_lfo_depth_oncc.iter() {
            let cc_val = self.cc_state[*cc as usize].load(Ordering::Relaxed) as f32 / 127.0;
            lfo_depth += delta * cc_val;
        }
        if lfo_freq > 0.0 && lfo_depth > 0.0 {
            // Advance phase by tick samples (~32). Wraps each cycle.
            let tick_samples = RECOMPUTE_INTERVAL as f32 * 4.0;
            self.lfo_phase += 2.0 * std::f32::consts::PI * lfo_freq * tick_samples
                / self.sample_rate;
            if self.lfo_phase > 2.0 * std::f32::consts::PI {
                self.lfo_phase -= 2.0 * std::f32::consts::PI;
            }
            cents_sum += self.lfo_phase.sin() * lfo_depth;
        }
        // MOVE FORK / Phase 11: filter envelope (autowah). Advances
        // one tick (~32 samples) at a time. Stage transitions when the
        // current ramp hits its target. `release` jumps the stage to R
        // as soon as fileg_released is set.
        if self.fileg_depth.abs() > 0.0 {
            let tick_secs = (RECOMPUTE_INTERVAL as f32 * 4.0) / self.sample_rate;
            if self.fileg_released && self.fileg_stage < 3 {
                self.fileg_stage = 3;
            }
            match self.fileg_stage {
                0 => {
                    let rate = if self.fileg_attack > 0.0 {
                        tick_secs / self.fileg_attack
                    } else { 1.0 };
                    self.fileg_level += rate;
                    if self.fileg_level >= 1.0 {
                        self.fileg_level = 1.0;
                        self.fileg_stage = 1;
                    }
                }
                1 => {
                    let rate = if self.fileg_decay > 0.0 {
                        tick_secs / self.fileg_decay
                    } else { 1.0 };
                    self.fileg_level -= rate * (1.0 - self.fileg_sustain);
                    if self.fileg_level <= self.fileg_sustain {
                        self.fileg_level = self.fileg_sustain;
                        self.fileg_stage = 2;
                    }
                }
                2 => {
                    // Sustain — hold level.
                }
                3 => {
                    let rate = if self.fileg_release > 0.0 {
                        tick_secs / self.fileg_release
                    } else { 1.0 };
                    self.fileg_level -= rate;
                    if self.fileg_level <= 0.0 {
                        self.fileg_level = 0.0;
                        self.fileg_stage = 4;
                    }
                }
                _ => {}
            }
            // MOVE FORK / 2026-05-16: when `fileg_curve` is set, route
            // the envelope level through a 128-point curve table before
            // multiplying by depth. Lets the SFZ converter pre-bake DS's
            // linear-Hz envelope sweep into the SFZ exp-cents domain.
            let env = if let Some(id) = self.fileg_curve {
                if let Some(table) = self.curves.get(&id) {
                    let idx = (self.fileg_level.clamp(0.0, 1.0) * 127.0).round() as usize;
                    table[idx.min(127)]
                } else {
                    self.fileg_level
                }
            } else {
                self.fileg_level
            };
            cents_sum += env * self.fileg_depth;
        }
        let mut db_sum = self.base_resonance_db;
        for (cc, delta) in self.resonance_oncc.iter() {
            let cc_val = self.cc_state[*cc as usize].load(Ordering::Relaxed);
            db_sum += delta * cc_lookup(cc_val, *cc, &self.resonance_curvecc, &self.curves);
        }
        let freq = self.base_freq * (cents_sum / 1200.0).exp2();
        let q = db_to_amp(db_sum) * Q_BUTTERWORTH_F32;
        (freq, q)
    }

    /// Advance one tick — sample target from CC, follow `current` toward
    /// it. Returns the smoothed `(freq, q)` to feed into biquad coeffs.
    #[inline(always)]
    fn step(&mut self) -> (f32, f32) {
        let (target_freq, target_q) = self.target();
        self.current_freq += (target_freq - self.current_freq) * self.smooth_alpha;
        self.current_q    += (target_q    - self.current_q)    * self.smooth_alpha;
        (self.current_freq, self.current_q)
    }

    /// Returns true iff coefficients should be updated this tick. When
    /// `true`, the caller MUST call `step()` to advance smoothing and
    /// derive new freq/Q before recomputing biquad coefficients.
    #[inline(always)]
    fn tick(&mut self) -> bool {
        if self.static_filter {
            return false;
        }
        if self.countdown == 0 {
            self.countdown = RECOMPUTE_INTERVAL;
            true
        } else {
            self.countdown -= 1;
            false
        }
    }
}

pub struct SIMDMonoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    v: V,
    cutoff: BiQuadFilter,
    _s: PhantomData<S>,
}

impl<S, V> SIMDMonoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    pub fn new(v: V, filter: &BiQuadFilter) -> Self {
        SIMDMonoVoiceCutoff {
            v,
            cutoff: filter.clone(),
            _s: PhantomData,
        }
    }
}

impl<S, V> VoiceGeneratorBase for SIMDMonoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    #[inline(always)]
    fn ended(&self) -> bool {
        self.v.ended()
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.v.signal_release(rel_type);
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.v.process_controls(control);
    }
}

impl<S, V> SIMDVoiceGenerator<S, SIMDSampleMono<S>> for SIMDMonoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleMono<S> {
        simd_invoke!(S, {
            let mut next_sample = self.v.next_sample();
            next_sample.0 = self.cutoff.process_simd::<S>(next_sample.0);
            next_sample
        })
    }
}

pub struct SIMDStereoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    v: V,
    cutoff1: BiQuadFilter,
    cutoff2: BiQuadFilter,
    _s: PhantomData<S>,
}

impl<S, V> SIMDStereoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    pub fn new(v: V, filter: &BiQuadFilter) -> Self {
        SIMDStereoVoiceCutoff {
            v,
            cutoff1: filter.clone(),
            cutoff2: filter.clone(),
            _s: PhantomData,
        }
    }
}

impl<S, V> VoiceGeneratorBase for SIMDStereoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    #[inline(always)]
    fn ended(&self) -> bool {
        self.v.ended()
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.v.signal_release(rel_type);
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.v.process_controls(control);
    }
}

impl<S, V> SIMDVoiceGenerator<S, SIMDSampleStereo<S>> for SIMDStereoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleStereo<S> {
        simd_invoke!(S, {
            let mut next_sample = self.v.next_sample();
            next_sample.0 = self.cutoff1.process_simd::<S>(next_sample.0);
            next_sample.1 = self.cutoff2.process_simd::<S>(next_sample.1);
            next_sample
        })
    }
}

/// MOVE FORK: mono cutoff with live `cutoff_oncc` / `resonance_oncc`
/// support. Recomputes biquad coefficients on a counter and applies
/// them to the held filter. When no bindings are present, the
/// `static_filter` short-circuit makes the cost identical to the
/// non-live SIMDMonoVoiceCutoff.
pub struct SIMDMonoVoiceCutoffLive<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    v: V,
    cutoff: BiQuadFilter,
    state: LiveCutoffState,
    _s: PhantomData<S>,
}

impl<S, V> SIMDMonoVoiceCutoffLive<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    pub fn new(
        v: V,
        filter: &BiQuadFilter,
        cc_state: CcState,
        cutoff_oncc: Arc<[(u8, f32)]>,
        resonance_oncc: Arc<[(u8, f32)]>,
        cutoff_curvecc: Arc<[(u8, u8)]>,
        resonance_curvecc: Arc<[(u8, u8)]>,
        curves: Arc<std::collections::HashMap<u8, [f32; 128]>>,
        fil_lfo_freq: f32,
        fil_lfo_depth: f32,
        fil_lfo_freq_oncc: Arc<[(u8, f32)]>,
        fil_lfo_depth_oncc: Arc<[(u8, f32)]>,
        fileg_attack: f32,
        fileg_decay: f32,
        fileg_sustain: f32,
        fileg_release: f32,
        fileg_depth: f32,
        fileg_curve: Option<u8>,
        fil_type: FilterType,
        sample_rate: f32,
        base_freq: f32,
        base_resonance_db: f32,
    ) -> Self {
        SIMDMonoVoiceCutoffLive {
            v,
            cutoff: filter.clone(),
            state: LiveCutoffState::new(
                cc_state,
                cutoff_oncc,
                resonance_oncc,
                cutoff_curvecc,
                resonance_curvecc,
                curves,
                fil_lfo_freq,
                fil_lfo_depth,
                fil_lfo_freq_oncc,
                fil_lfo_depth_oncc,
                fileg_attack,
                fileg_decay,
                fileg_sustain,
                fileg_release,
                fileg_depth,
                fileg_curve,
                fil_type,
                sample_rate,
                base_freq,
                base_resonance_db,
            ),
            _s: PhantomData,
        }
    }
}

impl<S, V> VoiceGeneratorBase for SIMDMonoVoiceCutoffLive<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    #[inline(always)]
    fn ended(&self) -> bool { self.v.ended() }
    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.state.notify_release();
        self.v.signal_release(rel_type);
    }
    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) { self.v.process_controls(control); }
}

impl<S, V> SIMDVoiceGenerator<S, SIMDSampleMono<S>> for SIMDMonoVoiceCutoffLive<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleMono<S> {
        if self.state.tick() {
            let (freq, q) = self.state.step();
            let coeffs = BiQuadFilter::get_coeffs(
                self.state.fil_type, freq, self.state.sample_rate, Some(q),
            );
            self.cutoff.set_coefficients(coeffs);
        }
        simd_invoke!(S, {
            let mut next_sample = self.v.next_sample();
            next_sample.0 = self.cutoff.process_simd::<S>(next_sample.0);
            next_sample
        })
    }
}

/// MOVE FORK: stereo cutoff with live `cutoff_oncc` / `resonance_oncc`.
pub struct SIMDStereoVoiceCutoffLive<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    v: V,
    cutoff1: BiQuadFilter,
    cutoff2: BiQuadFilter,
    state: LiveCutoffState,
    _s: PhantomData<S>,
}

impl<S, V> SIMDStereoVoiceCutoffLive<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    pub fn new(
        v: V,
        filter: &BiQuadFilter,
        cc_state: CcState,
        cutoff_oncc: Arc<[(u8, f32)]>,
        resonance_oncc: Arc<[(u8, f32)]>,
        cutoff_curvecc: Arc<[(u8, u8)]>,
        resonance_curvecc: Arc<[(u8, u8)]>,
        curves: Arc<std::collections::HashMap<u8, [f32; 128]>>,
        fil_lfo_freq: f32,
        fil_lfo_depth: f32,
        fil_lfo_freq_oncc: Arc<[(u8, f32)]>,
        fil_lfo_depth_oncc: Arc<[(u8, f32)]>,
        fileg_attack: f32,
        fileg_decay: f32,
        fileg_sustain: f32,
        fileg_release: f32,
        fileg_depth: f32,
        fileg_curve: Option<u8>,
        fil_type: FilterType,
        sample_rate: f32,
        base_freq: f32,
        base_resonance_db: f32,
    ) -> Self {
        SIMDStereoVoiceCutoffLive {
            v,
            cutoff1: filter.clone(),
            cutoff2: filter.clone(),
            state: LiveCutoffState::new(
                cc_state,
                cutoff_oncc,
                resonance_oncc,
                cutoff_curvecc,
                resonance_curvecc,
                curves,
                fil_lfo_freq,
                fil_lfo_depth,
                fil_lfo_freq_oncc,
                fil_lfo_depth_oncc,
                fileg_attack,
                fileg_decay,
                fileg_sustain,
                fileg_release,
                fileg_depth,
                fileg_curve,
                fil_type,
                sample_rate,
                base_freq,
                base_resonance_db,
            ),
            _s: PhantomData,
        }
    }
}

impl<S, V> VoiceGeneratorBase for SIMDStereoVoiceCutoffLive<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    #[inline(always)]
    fn ended(&self) -> bool { self.v.ended() }
    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.state.notify_release();
        self.v.signal_release(rel_type);
    }
    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) { self.v.process_controls(control); }
}

impl<S, V> SIMDVoiceGenerator<S, SIMDSampleStereo<S>> for SIMDStereoVoiceCutoffLive<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleStereo<S> {
        if self.state.tick() {
            let (freq, q) = self.state.step();
            let coeffs = BiQuadFilter::get_coeffs(
                self.state.fil_type, freq, self.state.sample_rate, Some(q),
            );
            self.cutoff1.set_coefficients(coeffs);
            self.cutoff2.set_coefficients(coeffs);
        }
        simd_invoke!(S, {
            let mut next_sample = self.v.next_sample();
            next_sample.0 = self.cutoff1.process_simd::<S>(next_sample.0);
            next_sample.1 = self.cutoff2.process_simd::<S>(next_sample.1);
            next_sample
        })
    }
}
