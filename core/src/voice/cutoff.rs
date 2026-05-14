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
/// modulators. Recomputes biquad coefficients from
/// `base_freq * 2^(Σ cents·cc/127/1200)` and
/// `db_to_amp(base_resonance_db + Σ db·cc/127) * Q_BUTTERWORTH_F32`
/// on a counter, then sets them on the held filter(s).
struct LiveCutoffState {
    cc_state: CcState,
    cutoff_oncc: Arc<[(u8, f32)]>,
    resonance_oncc: Arc<[(u8, f32)]>,
    fil_type: FilterType,
    sample_rate: f32,
    base_freq: f32,
    base_resonance_db: f32,
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
        Self {
            cc_state,
            cutoff_oncc,
            resonance_oncc,
            fil_type,
            sample_rate,
            base_freq,
            base_resonance_db,
            static_filter,
            countdown: 0,
        }
    }

    #[inline]
    fn compute(&self) -> (f32, f32) {
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

    /// Returns true iff coefficients should be updated this tick.
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
            let (freq, q) = self.state.compute();
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
            let (freq, q) = self.state.compute();
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
