// MOVE FORK: SIMD voice generator for live SFZ
// `pan_oncc<N>=<percent>` modulation. Replaces the static
// SIMDConstantStereo pan stage with a per-voice generator that polls
// the channel CC array on a counter and produces a stereo SIMD with
// cos/sin equal-power gains derived from `base_pan + Σ(delta·cc/127)`.
//
// Sampling cadence: every `RECOMPUTE_INTERVAL` calls to `next_sample`
// the bindings are reread from the CC atomics; in between, the cached
// stereo SIMD is reused. Matches voice/oncc_amp.rs cadence.
//
// Empty bindings short-circuit: when the region has no pan_oncc, the
// stereo gains are computed once at construction and reused forever —
// cost matches the static SIMDConstantStereo it replaces.

use std::sync::{atomic::Ordering, Arc};

use simdeez::prelude::*;

use crate::voice::{CcState, ReleaseType, VoiceControlData};

use super::{SIMDSampleStereo, SIMDVoiceGenerator, VoiceGeneratorBase};

const RECOMPUTE_INTERVAL: u32 = 8;

pub struct SIMDVoicePan<S: Simd> {
    cc_state: CcState,
    bindings: Arc<[(u8, f32)]>,
    curvecc: Arc<[(u8, u8)]>,
    curves: Arc<std::collections::HashMap<u8, [f32; 128]>>,
    /// Base pan in [-1, 1]. Per-CC deltas are in *percent* (SFZ
    /// pan_oncc is -100..100), converted to [-1, 1] when added.
    base_pan: f32,
    left_gain: S::Vf32,
    right_gain: S::Vf32,
    countdown: u32,
    static_pan: bool,
}

impl<S: Simd> SIMDVoicePan<S> {
    pub fn new(
        cc_state: CcState,
        bindings: Arc<[(u8, f32)]>,
        curvecc: Arc<[(u8, u8)]>,
        curves: Arc<std::collections::HashMap<u8, [f32; 128]>>,
        base_pan: f32,
    ) -> Self {
        let static_pan = bindings.is_empty();
        let pan = if static_pan {
            base_pan
        } else {
            compute_pan(&cc_state, &bindings, &curvecc, &curves, base_pan)
        };
        let (l, r) = gains(pan);
        simd_invoke!(S, {
            SIMDVoicePan {
                cc_state,
                bindings,
                curvecc,
                curves,
                base_pan,
                left_gain: S::Vf32::set1(l),
                right_gain: S::Vf32::set1(r),
                countdown: 0,
                static_pan,
            }
        })
    }
}

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
fn compute_pan(
    cc_state: &CcState,
    bindings: &[(u8, f32)],
    curvecc: &[(u8, u8)],
    curves: &std::collections::HashMap<u8, [f32; 128]>,
    base_pan: f32,
) -> f32 {
    let mut pan = base_pan;
    for (cc, delta_pct) in bindings.iter() {
        let cc_val = cc_state[*cc as usize].load(Ordering::Relaxed);
        pan += (*delta_pct * cc_lookup(cc_val, *cc, curvecc, curves)) / 100.0;
    }
    pan.clamp(-1.0, 1.0)
}

#[inline]
fn gains(pan: f32) -> (f32, f32) {
    let p = pan * std::f32::consts::PI / 2.0;
    let l = (p.cos() * 1.42).min(1.0);
    let r = (p.sin() * 1.42).min(1.0);
    (l, r)
}

impl<S: Simd> VoiceGeneratorBase for SIMDVoicePan<S> {
    #[inline(always)]
    fn ended(&self) -> bool {
        false
    }

    #[inline(always)]
    fn signal_release(&mut self, _rel_type: ReleaseType) {}

    #[inline(always)]
    fn process_controls(&mut self, _control: &VoiceControlData) {}
}

impl<S: Simd> SIMDVoiceGenerator<S, SIMDSampleStereo<S>> for SIMDVoicePan<S> {
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleStereo<S> {
        if !self.static_pan {
            if self.countdown == 0 {
                let pan = compute_pan(
                    &self.cc_state,
                    &self.bindings,
                    &self.curvecc,
                    &self.curves,
                    self.base_pan,
                );
                let (l, r) = gains(pan);
                simd_invoke!(S, {
                    self.left_gain = S::Vf32::set1(l);
                    self.right_gain = S::Vf32::set1(r);
                });
                self.countdown = RECOMPUTE_INTERVAL;
            } else {
                self.countdown -= 1;
            }
        }
        SIMDSampleStereo(self.left_gain, self.right_gain)
    }
}
