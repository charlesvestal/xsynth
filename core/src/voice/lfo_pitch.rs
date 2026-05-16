// MOVE FORK / Phase 11: per-voice pitch LFO (vibrato). Outputs a
// SIMDSampleMono whose value is the pitch-multiplier factor — combined
// (multiplied) into the sampler's pitch_fac chain so the sampler reads
// samples at `base_speed * pitch_mod`.
//
// Cents → factor: `2^(sin·depth_cents/1200)`. Depth 100 cents = ±5.9%
// pitch swing (one semitone), so 7Hz/100cents gives a normal vibrato.
//
// Sampling cadence matches SIMDVoiceLfoAmp — phase advances every
// next_sample call, the factor is recomputed every RECOMPUTE_INTERVAL
// vectors. Pitch perception is dominated by ~5-7Hz wobble; the coarse
// sampling cadence (~32 samples = ~0.7ms) is well below audible.

use std::sync::{atomic::Ordering, Arc};

use simdeez::prelude::*;

use crate::voice::{CcState, ReleaseType, VoiceControlData};

use super::{SIMDSampleMono, SIMDVoiceGenerator, VoiceGeneratorBase};

const RECOMPUTE_INTERVAL: u32 = 8;

pub struct SIMDVoiceLfoPitch<S: Simd> {
    cc_state: CcState,
    freq_oncc: Arc<[(u8, f32)]>,
    depth_oncc: Arc<[(u8, f32)]>,
    sample_rate: f32,
    base_freq: f32,
    base_depth_cents: f32,
    phase: f32,
    values: S::Vf32,
    countdown: u32,
    static_pitch: bool,
}

impl<S: Simd> SIMDVoiceLfoPitch<S> {
    pub fn new(
        cc_state: CcState,
        freq_hz: f32,
        depth_cents: f32,
        sample_rate: f32,
        freq_oncc: Arc<[(u8, f32)]>,
        depth_oncc: Arc<[(u8, f32)]>,
    ) -> Self {
        let max_freq_delta: f32 = freq_oncc.iter().map(|(_, d)| d.abs()).sum();
        let max_depth_delta: f32 = depth_oncc.iter().map(|(_, d)| d.abs()).sum();
        let static_pitch = (freq_hz + max_freq_delta) <= 0.0
            || (depth_cents.abs() + max_depth_delta) <= 0.0;
        simd_invoke!(S, {
            SIMDVoiceLfoPitch {
                cc_state,
                freq_oncc,
                depth_oncc,
                sample_rate,
                base_freq: freq_hz,
                base_depth_cents: depth_cents,
                phase: 0.0,
                values: S::Vf32::set1(1.0),
                countdown: 0,
                static_pitch,
            }
        })
    }

    #[inline]
    fn live_params(&self) -> (f32, f32) {
        let mut freq = self.base_freq;
        for (cc, delta) in self.freq_oncc.iter() {
            let v = self.cc_state[*cc as usize].load(Ordering::Relaxed) as f32 / 127.0;
            freq += delta * v;
        }
        let mut depth = self.base_depth_cents;
        for (cc, delta) in self.depth_oncc.iter() {
            let v = self.cc_state[*cc as usize].load(Ordering::Relaxed) as f32 / 127.0;
            depth += delta * v;
        }
        if freq < 0.0 { freq = 0.0; }
        (freq, depth)
    }
}

impl<S: Simd> VoiceGeneratorBase for SIMDVoiceLfoPitch<S> {
    #[inline(always)]
    fn ended(&self) -> bool { false }
    #[inline(always)]
    fn signal_release(&mut self, _rel_type: ReleaseType) {}
    #[inline(always)]
    fn process_controls(&mut self, _control: &VoiceControlData) {}
}

impl<S: Simd> SIMDVoiceGenerator<S, SIMDSampleMono<S>> for SIMDVoiceLfoPitch<S> {
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleMono<S> {
        if !self.static_pitch {
            let (freq, depth_cents) = self.live_params();
            let phase_inc = 2.0 * std::f32::consts::PI * freq / self.sample_rate;
            self.phase += phase_inc * 4.0;
            if self.phase > 2.0 * std::f32::consts::PI {
                self.phase -= 2.0 * std::f32::consts::PI;
            }
            if self.countdown == 0 {
                let factor = if freq > 0.0 && depth_cents.abs() > 0.0 {
                    (self.phase.sin() * depth_cents / 1200.0).exp2()
                } else {
                    1.0
                };
                simd_invoke!(S, {
                    self.values = S::Vf32::set1(factor);
                });
                self.countdown = RECOMPUTE_INTERVAL;
            } else {
                self.countdown -= 1;
            }
        }
        SIMDSampleMono(self.values)
    }
}
