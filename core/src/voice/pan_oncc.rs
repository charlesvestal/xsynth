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
//
// MOVE FORK / Phase 11: pan LFO source piggybacks on the same generator.
// `pan_lfo_freq` (Hz) + `pan_lfo_depth` (percent of pan range) drive a
// sine that adds into the pan offset just like the oncc bindings.

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
    /// MOVE FORK / Phase 11: pan LFO state. Active when freq>0 and
    /// depth>0 (after live CC contributions).
    sample_rate: f32,
    pan_lfo_freq: f32,
    pan_lfo_depth: f32,
    pan_lfo_freq_oncc: Arc<[(u8, f32)]>,
    pan_lfo_depth_oncc: Arc<[(u8, f32)]>,
    lfo_phase: f32,
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
        sample_rate: f32,
        pan_lfo_freq: f32,
        pan_lfo_depth: f32,
        pan_lfo_freq_oncc: Arc<[(u8, f32)]>,
        pan_lfo_depth_oncc: Arc<[(u8, f32)]>,
    ) -> Self {
        // A region with no oncc + no LFO is fully static.
        let max_lfo_freq_delta: f32 = pan_lfo_freq_oncc.iter().map(|(_, d)| d.abs()).sum();
        let max_lfo_depth_delta: f32 = pan_lfo_depth_oncc.iter().map(|(_, d)| d.abs()).sum();
        let has_lfo = (pan_lfo_freq + max_lfo_freq_delta) > 0.0
            && (pan_lfo_depth + max_lfo_depth_delta) > 0.0;
        let static_pan = bindings.is_empty() && !has_lfo;
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
                sample_rate,
                pan_lfo_freq,
                pan_lfo_depth,
                pan_lfo_freq_oncc,
                pan_lfo_depth_oncc,
                lfo_phase: 0.0,
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
                let mut pan = compute_pan(
                    &self.cc_state,
                    &self.bindings,
                    &self.curvecc,
                    &self.curves,
                    self.base_pan,
                );
                // Phase 11: pan LFO. Live freq + depth from base + oncc.
                let mut lfo_freq = self.pan_lfo_freq;
                for (cc, d) in self.pan_lfo_freq_oncc.iter() {
                    let v = self.cc_state[*cc as usize].load(Ordering::Relaxed) as f32 / 127.0;
                    lfo_freq += d * v;
                }
                let mut lfo_depth = self.pan_lfo_depth;
                for (cc, d) in self.pan_lfo_depth_oncc.iter() {
                    let v = self.cc_state[*cc as usize].load(Ordering::Relaxed) as f32 / 127.0;
                    lfo_depth += d * v;
                }
                if lfo_freq > 0.0 && lfo_depth > 0.0 {
                    let tick_samples = RECOMPUTE_INTERVAL as f32 * 4.0;
                    self.lfo_phase += 2.0 * std::f32::consts::PI * lfo_freq
                                      * tick_samples / self.sample_rate;
                    if self.lfo_phase > 2.0 * std::f32::consts::PI {
                        self.lfo_phase -= 2.0 * std::f32::consts::PI;
                    }
                    pan += (self.lfo_phase.sin() * lfo_depth) / 100.0;
                    if pan < -1.0 { pan = -1.0; } else if pan > 1.0 { pan = 1.0; }
                }
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
