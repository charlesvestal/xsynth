// MOVE FORK / Phase 12: stereo chorus on the channel post-mix using
// fundsp's `chorus()` AudioUnit.
//
// Each side runs its own `chorus()` with a different seed for stereo
// width (fundsp randomizes LFO phases / delay tap positions per seed).
// `mix` is the channel-side dry/wet crossfade (DS FX_MIX convention).
//
// Parameter updates (rate / depth) rebuild both AudioUnits — fundsp's
// chorus takes its mod_frequency / variation at construction. Knob
// sweeps cause brief discontinuities; acceptable until we plumb
// Shared<f32> values through the graph.

use fundsp::prelude::AudioUnit;
use fundsp::hacker32::chorus;

pub struct StereoChorus {
    sample_rate: f32,
    rate: f32,
    depth: f32,
    mix: f32,
    left: Box<dyn AudioUnit + Send>,
    right: Box<dyn AudioUnit + Send>,
}

impl StereoChorus {
    pub fn new(sample_rate: f32) -> Self {
        let rate = 0.7_f32;
        let depth = 0.5_f32;
        let mut left = build_unit(rate, depth, 0x11);
        let mut right = build_unit(rate, depth, 0x22);
        left.set_sample_rate(sample_rate as f64);
        right.set_sample_rate(sample_rate as f64);
        Self {
            sample_rate,
            rate,
            depth,
            mix: 0.0,
            left,
            right,
        }
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

    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    pub fn clear(&mut self) {
        self.left.reset();
        self.right.reset();
    }

    fn rebuild(&mut self) {
        self.left = build_unit(self.rate, self.depth, 0x11);
        self.right = build_unit(self.rate, self.depth, 0x22);
        self.left.set_sample_rate(self.sample_rate as f64);
        self.right.set_sample_rate(self.sample_rate as f64);
    }

    /// Process one stereo frame in-place. Returns the (out_l, out_r).
    /// fundsp `chorus` takes 1 input, gives 1 output — we run one per
    /// side independently.
    #[inline]
    pub fn process(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let input_l = [in_l];
        let input_r = [in_r];
        let mut out_l = [0.0_f32];
        let mut out_r = [0.0_f32];
        self.left.tick(&input_l, &mut out_l);
        self.right.tick(&input_r, &mut out_r);
        // Crossfade dry/wet (FX_MIX convention).
        let dry = 1.0 - self.mix;
        let wet = self.mix;
        (in_l * dry + out_l[0] * wet, in_r * dry + out_r[0] * wet)
    }
}

fn build_unit(rate: f32, depth: f32, seed: u64) -> Box<dyn AudioUnit + Send> {
    // fundsp chorus signature: chorus(seed, separation, variation, mod_frequency).
    // separation = base delay separation (seconds).
    // variation = mod depth amplitude.
    // mod_frequency = LFO Hz.
    //
    // We use a base separation of 0.015 s (15 ms) which sits in the
    // classic chorus range. depth maps to variation 0..0.015 (full
    // swing matches the base delay, gentle chorus).
    let separation = 0.015_f32;
    let variation = (depth * 0.015).max(0.0001);
    Box::new(chorus(seed, separation, variation, rate))
}
