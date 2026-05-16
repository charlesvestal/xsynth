// MOVE FORK / Phase 12: stereo phaser on the channel post-mix using
// fundsp's `phaser()` AudioUnit.
//
// Same pattern as StereoChorus — two independent units, distinct LFO
// phase offsets between L/R for stereo movement. fundsp's phaser takes
// `feedback_amount` and a `|t| -> 0..1` modulation closure at
// construction, so rate/depth/feedback updates rebuild the units.
//
// FX_CENTER_FREQUENCY from DS is not directly honored — fundsp's
// allpass-cascade phaser has an internal center; the modulation closure
// sweeps it. Author center-freq settings are silently dropped (the
// audible difference is small relative to depth/rate/feedback).

use fundsp::prelude::AudioUnit;
use fundsp::hacker32::{phaser, sin_hz};

pub struct StereoPhaser {
    sample_rate: f32,
    rate: f32,
    depth: f32,
    feedback: f32,
    mix: f32,
    left: Box<dyn AudioUnit + Send>,
    right: Box<dyn AudioUnit + Send>,
}

impl StereoPhaser {
    pub fn new(sample_rate: f32) -> Self {
        let rate = 0.4_f32;
        let depth = 0.5_f32;
        let feedback = 0.5_f32;
        let mut left  = build_unit(rate, depth, feedback, 0.0);
        let mut right = build_unit(rate, depth, feedback, 0.5);
        left.set_sample_rate(sample_rate as f64);
        right.set_sample_rate(sample_rate as f64);
        Self { sample_rate, rate, depth, feedback, mix: 0.0, left, right }
    }

    pub fn set_rate(&mut self, rate: f32) {
        let r = rate.clamp(0.05, 12.0);
        if (r - self.rate).abs() < 1e-4 { return; }
        self.rate = r;
        self.rebuild();
    }

    pub fn set_depth(&mut self, depth: f32) {
        let d = depth.clamp(0.0, 1.0);
        if (d - self.depth).abs() < 1e-3 { return; }
        self.depth = d;
        self.rebuild();
    }

    pub fn set_feedback(&mut self, fb: f32) {
        let f = fb.clamp(0.0, 0.95);
        if (f - self.feedback).abs() < 1e-3 { return; }
        self.feedback = f;
        self.rebuild();
    }

    pub fn set_mix(&mut self, mix: f32) { self.mix = mix.clamp(0.0, 1.0); }

    pub fn clear(&mut self) {
        self.left.reset();
        self.right.reset();
    }

    fn rebuild(&mut self) {
        self.left  = build_unit(self.rate, self.depth, self.feedback, 0.0);
        self.right = build_unit(self.rate, self.depth, self.feedback, 0.5);
        self.left.set_sample_rate(self.sample_rate as f64);
        self.right.set_sample_rate(self.sample_rate as f64);
    }

    /// Process one stereo frame. Returns (out_l, out_r) crossfaded with
    /// dry per `mix` (FX_MIX convention).
    #[inline]
    pub fn process(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let input_l = [in_l];
        let input_r = [in_r];
        let mut out_l = [0.0_f32];
        let mut out_r = [0.0_f32];
        self.left.tick(&input_l, &mut out_l);
        self.right.tick(&input_r, &mut out_r);
        let dry = 1.0 - self.mix;
        let wet = self.mix;
        (in_l * dry + out_l[0] * wet, in_r * dry + out_r[0] * wet)
    }
}

fn build_unit(rate: f32, depth: f32, feedback: f32, phase_offset: f32)
    -> Box<dyn AudioUnit + Send>
{
    // depth scales the modulation range. fundsp phaser closure must
    // return 0..1; we center the sin output around 0.5 and scale by
    // depth so 0 = no mod, 1 = full 0..1 sweep. phase_offset shifts
    // L vs R for stereo movement.
    let r = rate;
    let d = depth;
    let off = phase_offset;
    Box::new(phaser(feedback, move |t| {
        // sin_hz returns -1..1; map to 0..1 then scale by depth around
        // 0.5 so we always stay inside the closure's required range.
        let s = sin_hz(r, t + off);
        0.5 + 0.5 * d * s
    }))
}
