// MOVE FORK / Phase 9: stereo feedback delay on the channel post-mix.
//
// Hand-rolled ring buffer per channel (fundsp's compile-time graph
// makes runtime feedback updates awkward). At ~2s max delay × 2
// channels × 4 bytes per f32, the buffer is ~1.4 MB per channel which
// is fine — allocated once at construction.
//
// Per-channel state: ring buffer of MAX_LEN samples + write head.
// `process(sample)` reads from `write_head - delay_samples`, mixes
// it back via `feedback`, writes the new sum to the head, advances.
//
// Channel applies dry/wet `mix` outside this struct.

const MAX_DELAY_SECS: usize = 2;

pub struct StereoFeedbackDelay {
    sample_rate: f32,
    delay_samples: usize,
    feedback: f32,
    /// Per-channel ring buffer; size = MAX_DELAY_SECS * sample_rate
    /// rounded up to next power of two so the wrap can be masked.
    /// For simplicity we just use modulo since size doesn't need to
    /// be 2^N.
    buf_l: Vec<f32>,
    buf_r: Vec<f32>,
    write_head: usize,
}

impl StereoFeedbackDelay {
    pub fn new(sample_rate: f32) -> Self {
        let cap = (MAX_DELAY_SECS as f32 * sample_rate).ceil() as usize;
        Self {
            sample_rate,
            delay_samples: (sample_rate as usize).max(1) / 4,  // 0.25 s default
            feedback: 0.4,
            buf_l: vec![0.0; cap],
            buf_r: vec![0.0; cap],
            write_head: 0,
        }
    }

    /// Set delay time in seconds (clamped to MAX_DELAY_SECS).
    pub fn set_time(&mut self, seconds: f32) {
        let s = seconds.clamp(0.001, MAX_DELAY_SECS as f32);
        self.delay_samples =
            ((s * self.sample_rate) as usize).max(1).min(self.buf_l.len() - 1);
    }

    pub fn set_feedback(&mut self, fb: f32) {
        self.feedback = fb.clamp(0.0, 0.95);
    }

    /// Process one stereo frame in-place. Returns the wet sample only
    /// (the channel mixes dry/wet outside).
    #[inline]
    pub fn process(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let cap = self.buf_l.len();
        let read_head = if self.write_head >= self.delay_samples {
            self.write_head - self.delay_samples
        } else {
            cap - (self.delay_samples - self.write_head)
        };
        let wet_l = self.buf_l[read_head];
        let wet_r = self.buf_r[read_head];
        // Standard feedback-delay topology: write = input + feedback * wet.
        self.buf_l[self.write_head] = in_l + wet_l * self.feedback;
        self.buf_r[self.write_head] = in_r + wet_r * self.feedback;
        self.write_head = (self.write_head + 1) % cap;
        (wet_l, wet_r)
    }

    /// Clear the ring buffer — call on instrument switch so the
    /// previous preset's tail doesn't leak into the next.
    pub fn clear(&mut self) {
        for s in self.buf_l.iter_mut() { *s = 0.0; }
        for s in self.buf_r.iter_mut() { *s = 0.0; }
        self.write_head = 0;
    }
}
