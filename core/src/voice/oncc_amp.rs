// MOVE FORK: SIMD voice generator implementing live SFZ
// `volume_oncc<N>=<dB>` modulation.
//
// Multiplies the voice amp by `db_to_amp(Σ delta_db · cc<n>/127)`
// across the region's bindings. Each binding is (cc_number, dB-delta).
//
// Sampling cadence: the cached `set1(amp)` is recomputed every
// `RECOMPUTE_INTERVAL` calls to `next_sample`. Atomic loads + powf
// happen at that cadence — every other call is a register move.
// At 44.1 kHz with NEON's S::Vf32::WIDTH = 4 and interval = 8, we
// recompute ~1.4k times/sec per voice — well below human-audible
// stepping (~25 ms granularity feels continuous on knob sweeps) and
// trivial CPU cost (one atomic read + one powf per voice per ~32
// audio samples).
//
// Empty bindings short-circuit: `static_amp` flag lets the generator
// skip all atomic loads and powf calls, returning a permanent
// `set1(1.0)`. Cost matches the existing velocity SIMDConstant stage —
// one SIMD multiply per sample-width.
//
// We DON'T fire `propagate_voice_controls` on Raw CC events; the
// previous Phase 3 attempt did and that pinned the audio thread on
// heavy patches during knob sweeps. The atomic-poll approach is
// bounded and predictable instead.

use std::sync::{atomic::Ordering, Arc};

use simdeez::prelude::*;

use crate::{
    helpers::db_to_amp,
    voice::{CcState, ReleaseType, VoiceControlData},
};

use super::{SIMDSampleMono, SIMDVoiceGenerator, VoiceGeneratorBase};

/// `next_sample` calls between atomic CC reads. With S::Vf32::WIDTH=4
/// on aarch64 NEON, the per-voice resample rate is ~1/32 audio
/// samples ≈ 1.4 kHz at 44.1 kHz output. Lower values give smoother
/// knob sweeps at the cost of more atomic loads + powf per voice.
const RECOMPUTE_INTERVAL: u32 = 8;

pub struct SIMDVoiceOnccAmp<S: Simd> {
    cc_state: CcState,
    bindings: Arc<[(u8, f32)]>,
    /// MOVE FORK / Phase 6: per-CC curve_id (when a `volume_curvecc<N>`
    /// references one). The SIMD generator uses `curves[id][cc]` instead
    /// of `cc/127` when present. Empty when the region has no curvecc.
    curvecc: Arc<[(u8, u8)]>,
    curves: Arc<std::collections::HashMap<u8, [f32; 128]>>,
    values: S::Vf32,
    countdown: u32,
    /// True when bindings is empty — we cache `set1(1.0)` once at
    /// construction and never touch atomics again. Lets non-`_oncc`
    /// regions pay only one SIMD multiply per sample-width.
    static_amp: bool,
}

impl<S: Simd> SIMDVoiceOnccAmp<S> {
    pub fn new(
        cc_state: CcState,
        bindings: Arc<[(u8, f32)]>,
        curvecc: Arc<[(u8, u8)]>,
        curves: Arc<std::collections::HashMap<u8, [f32; 128]>>,
    ) -> Self {
        let static_amp = bindings.is_empty();
        let amp = if static_amp {
            1.0
        } else {
            compute_amp(&cc_state, &bindings, &curvecc, &curves)
        };
        simd_invoke!(S, {
            SIMDVoiceOnccAmp {
                cc_state,
                bindings,
                curvecc,
                curves,
                values: S::Vf32::set1(amp),
                countdown: 0,
                static_amp,
            }
        })
    }
}

/// MOVE FORK / Phase 6: resolve the CC lookup. When the binding's CC
/// has a matching `_curvecc<N>=<id>` entry AND the curve table exists,
/// return `curves[id][cc_val]`; otherwise the linear `cc_val/127`.
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

#[inline]
fn compute_amp(
    cc_state: &CcState,
    bindings: &[(u8, f32)],
    curvecc: &[(u8, u8)],
    curves: &std::collections::HashMap<u8, [f32; 128]>,
) -> f32 {
    let mut db = 0.0_f32;
    for (cc, delta_db) in bindings.iter() {
        let cc_val = cc_state[*cc as usize].load(Ordering::Relaxed);
        db += delta_db * cc_lookup(cc_val, *cc, curvecc, curves);
    }
    db_to_amp(db)
}

impl<S: Simd> VoiceGeneratorBase for SIMDVoiceOnccAmp<S> {
    #[inline(always)]
    fn ended(&self) -> bool {
        false
    }

    #[inline(always)]
    fn signal_release(&mut self, _rel_type: ReleaseType) {}

    /// No-op — atomic resampling happens in `next_sample` on a counter.
    /// We don't depend on the channel firing propagate on Raw CCs
    /// (which would deadlock heavy patches under knob sweep load).
    #[inline(always)]
    fn process_controls(&mut self, _control: &VoiceControlData) {}
}

impl<S: Simd> SIMDVoiceGenerator<S, SIMDSampleMono<S>> for SIMDVoiceOnccAmp<S> {
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleMono<S> {
        if !self.static_amp {
            if self.countdown == 0 {
                let amp = compute_amp(&self.cc_state, &self.bindings, &self.curvecc, &self.curves);
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
