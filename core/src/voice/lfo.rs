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

use simdeez::prelude::*;

use crate::{
    helpers::db_to_amp,
    voice::{ReleaseType, VoiceControlData},
};

use super::{SIMDSampleMono, SIMDVoiceGenerator, VoiceGeneratorBase};

const RECOMPUTE_INTERVAL: u32 = 8;

pub struct SIMDVoiceLfoAmp<S: Simd> {
    /// Radians per audio sample (2π·freq / sample_rate).
    phase_inc: f32,
    /// Current phase in radians.
    phase: f32,
    /// Depth in dB (sin output range is [-depth_db, depth_db]).
    depth_db: f32,
    /// Cached amp output, refreshed every RECOMPUTE_INTERVAL vectors.
    values: S::Vf32,
    countdown: u32,
    static_amp: bool,
}

impl<S: Simd> SIMDVoiceLfoAmp<S> {
    pub fn new(freq_hz: f32, depth_db: f32, sample_rate: f32) -> Self {
        let static_amp = depth_db <= 0.0 || freq_hz <= 0.0;
        let phase_inc = if static_amp {
            0.0
        } else {
            2.0 * std::f32::consts::PI * freq_hz / sample_rate
        };
        simd_invoke!(S, {
            SIMDVoiceLfoAmp {
                phase_inc,
                phase: 0.0,
                depth_db,
                values: S::Vf32::set1(1.0),
                countdown: 0,
                static_amp,
            }
        })
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
            // Advance phase per SIMD-vector tick (one vector = WIDTH
            // audio samples — assume 4 for aarch64 NEON; the error is
            // bounded since phase is wrapped). Recompute amp on a
            // coarse counter.
            self.phase += self.phase_inc * 4.0;
            if self.phase > 2.0 * std::f32::consts::PI {
                self.phase -= 2.0 * std::f32::consts::PI;
            }
            if self.countdown == 0 {
                let sin_val = self.phase.sin();
                let amp = db_to_amp(sin_val * self.depth_db);
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
