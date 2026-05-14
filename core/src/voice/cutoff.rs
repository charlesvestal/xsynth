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
        fil_type: FilterType,
        sample_rate: f32,
        base_freq: f32,
        base_resonance_db: f32,
    ) -> Self {
        let static_filter = cutoff_oncc.is_empty() && resonance_oncc.is_empty();
        // 10ms smoothing time constant. tick_samples = RECOMPUTE_INTERVAL
        // * a SIMD vector width (assume 4 for aarch64 NEON; the exact
        // value isn't critical — alpha just controls smoothing speed,
        // any reasonable SIMD width gives a similar audible time
        // constant).
        let tick_samples = (RECOMPUTE_INTERVAL as f32) * 4.0;
        let smooth_samples = sample_rate * 0.010;
        let smooth_alpha = 1.0 - (-tick_samples / smooth_samples).exp();
        let init_q = db_to_amp(base_resonance_db) * Q_BUTTERWORTH_F32;
        Self {
            cc_state,
            cutoff_oncc,
            resonance_oncc,
            fil_type,
            sample_rate,
            base_freq,
            base_resonance_db,
            current_freq: base_freq,
            current_q: init_q,
            smooth_alpha,
            static_filter,
            countdown: 0,
        }
    }

    /// Target (freq, Q) from the live CC state.
    #[inline]
    fn target(&self) -> (f32, f32) {
        let mut cents_sum = 0.0_f32;
        for (cc, delta) in self.cutoff_oncc.iter() {
            let v = self.cc_state[*cc as usize].load(Ordering::Relaxed) as f32 / 127.0;
            cents_sum += delta * v;
        }
        let mut db_sum = self.base_resonance_db;
        for (cc, delta) in self.resonance_oncc.iter() {
            let v = self.cc_state[*cc as usize].load(Ordering::Relaxed) as f32 / 127.0;
            db_sum += delta * v;
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
    fn signal_release(&mut self, rel_type: ReleaseType) { self.v.signal_release(rel_type); }
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
    fn signal_release(&mut self, rel_type: ReleaseType) { self.v.signal_release(rel_type); }
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
