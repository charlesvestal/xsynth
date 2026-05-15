// MOVE FORK / Phase 11: per-voice sine LFO modulating voice amp
// (tremolo). Always-inserted SIMD stage with a `static_amp`
// short-circuit when `depth_db == 0` so non-modulated voices pay
// only one SIMD multiply per sample-width — same cost as the
// SIMDConstant velocity stage.
//
// At depth_db > 0, the LFO output `sin(2π·freq·t)` ∈ [-1, 1] is
// converted to a gain via `db_to_amp(sin · depth_db)`. So depth=6
// → ±6 dB swing; depth=20 → ±20 dB.
//
// Phase advance is per audio sample, but the cached `set1(amp)` is
// recomputed every `RECOMPUTE_INTERVAL` SIMD vectors (matches the
// other oncc generators). The phase still advances every sample so
// the LFO stays in sync — we just sample the resulting amp at
// coarser granularity to save sin() / powf() calls.

use std::sync::{atomic::Ordering, Arc};

use simdeez::prelude::*;

use crate::{
    helpers::db_to_amp,
    voice::{CcState, ReleaseType, VoiceControlData},
};

use super::{SIMDSampleMono, SIMDVoiceGenerator, VoiceGeneratorBase};

const RECOMPUTE_INTERVAL: u32 = 8;

pub struct SIMDVoiceLfoAmp<S: Simd> {
    cc_state: CcState,
    /// Live CC routing for freq (Hz delta added at cc/127).
    freq_oncc: Arc<[(u8, f32)]>,
    /// Live CC routing for depth (dB delta added at cc/127).
    depth_oncc: Arc<[(u8, f32)]>,
    sample_rate: f32,
    base_freq: f32,
    base_depth_db: f32,
    /// Current phase in radians.
    phase: f32,
    /// Cached amp output, refreshed every RECOMPUTE_INTERVAL vectors.
    values: S::Vf32,
    countdown: u32,
    /// True when the LFO can never become active (base + max CC delta
    /// keeps freq or depth at 0). Cheap short-circuit.
    static_amp: bool,
}

impl<S: Simd> SIMDVoiceLfoAmp<S> {
    pub fn new(
        cc_state: CcState,
        freq_hz: f32,
        depth_db: f32,
        sample_rate: f32,
        freq_oncc: Arc<[(u8, f32)]>,
        depth_oncc: Arc<[(u8, f32)]>,
    ) -> Self {
        // A region with no oncc routing AND base freq/depth both
        // zero can never produce tremolo. With oncc routing the LFO
        // becomes live as soon as a CC moves.
        let max_freq_delta: f32 = freq_oncc.iter().map(|(_, d)| d.abs()).sum();
        let max_depth_delta: f32 = depth_oncc.iter().map(|(_, d)| d.abs()).sum();
        let static_amp = (freq_hz + max_freq_delta) <= 0.0
            || (depth_db + max_depth_delta) <= 0.0;
        simd_invoke!(S, {
            SIMDVoiceLfoAmp {
                cc_state,
                freq_oncc,
                depth_oncc,
                sample_rate,
                base_freq: freq_hz,
                base_depth_db: depth_db,
                phase: 0.0,
                values: S::Vf32::set1(1.0),
                countdown: 0,
                static_amp,
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
        let mut depth = self.base_depth_db;
        for (cc, delta) in self.depth_oncc.iter() {
            let v = self.cc_state[*cc as usize].load(Ordering::Relaxed) as f32 / 127.0;
            depth += delta * v;
        }
        if freq < 0.0 { freq = 0.0; }
        if depth < 0.0 { depth = 0.0; }
        (freq, depth)
    }
}

impl<S: Simd> VoiceGeneratorBase for SIMDVoiceLfoAmp<S> {
    #[inline(always)]
    fn ended(&self) -> bool { false }
    #[inline(always)]
    fn signal_release(&mut self, _rel_type: ReleaseType) {}
    #[inline(always)]
    fn process_controls(&mut self, _control: &VoiceControlData) {}
}

impl<S: Simd> SIMDVoiceGenerator<S, SIMDSampleMono<S>> for SIMDVoiceLfoAmp<S> {
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleMono<S> {
        if !self.static_amp {
            let (freq, depth_db) = self.live_params();
            // Advance phase per SIMD-vector tick (4 samples assumed).
            let phase_inc = 2.0 * std::f32::consts::PI * freq / self.sample_rate;
            self.phase += phase_inc * 4.0;
            if self.phase > 2.0 * std::f32::consts::PI {
                self.phase -= 2.0 * std::f32::consts::PI;
            }
            if self.countdown == 0 {
                let amp = if depth_db > 0.0 && freq > 0.0 {
                    db_to_amp(self.phase.sin() * depth_db)
                } else {
                    1.0
                };
                simd_invoke!(S, {
                    self.values = S::Vf32::set1(amp);
                });
                self.countdown = RECOMPUTE_INTERVAL;
            } else {
                self.countdown -= 1;
            }
        }
        SIMDSampleMono(self.values)
    }
}
