use std::sync::{atomic::{AtomicU32, AtomicU64, Ordering}, Arc};
use std::time::Instant;

// MOVE FORK: per-block render breakdown — diagnostic only. Statics race
// across channels but for one active SFZ instance the values are usable.
// Read via `take_render_breakdown_us` and logged from the plugin's
// render path to localize where overhead hides at low voice counts.
pub static LAST_PARALLEL_US:   AtomicU32 = AtomicU32::new(0);
pub static LAST_SUM_US:        AtomicU32 = AtomicU32::new(0);
pub static LAST_FX_US:         AtomicU32 = AtomicU32::new(0);
pub static LAST_TOTAL_US:      AtomicU32 = AtomicU32::new(0);

use crate::{
    effects::MultiChannelBiQuad,
    helpers::{prepapre_cache_vec, sum_simd},
    voice::{new_cc_state, CcState, VoiceControlData},
    AudioStreamParams, ChannelCount,
};

use xsynth_soundfonts::FilterType;

use self::{control::ControlEventData, key::KeyData, params::VoiceChannelParams};

use super::AudioPipe;

use rayon::prelude::*;

mod channel_sf;
mod control;
mod key;
mod params;
mod voice_buffer;
mod voice_spawner;

mod event;
pub use event::*;
pub use channel_sf::SoundfontDropSink;

pub(crate) use control::ValueLerp;
pub use params::VoiceChannelStatsReader;

struct Key {
    data: KeyData,
    audio_cache: Vec<f32>,
    event_cache: Vec<KeyNoteEvent>,
}

/// MOVE FORK: drain a key's event cache, honoring a shared spawn budget.
/// `budget` is None when the channel has no spawn-burst limit (drain
/// everything). When Some, each NoteOn CAS-decrements the budget; if no
/// slot is available, the remaining events stay in event_cache (in their
/// original order) for the next block. Non-NoteOn events are always
/// drained — they don't spawn voices.
fn drain_events_with_budget(
    key: &mut Key,
    budget: Option<&Arc<std::sync::atomic::AtomicI64>>,
    control_data: &VoiceControlData,
    cc_state: &CcState,
    channel_sf: &channel_sf::ChannelSoundfont,
    layers: Option<usize>,
) {
    use std::sync::atomic::Ordering;
    let mut consumed = 0usize;
    if let Some(budget) = budget {
        for e in key.event_cache.iter() {
            if matches!(e, KeyNoteEvent::On(_)) {
                let mut got_slot = false;
                loop {
                    let prev = budget.load(Ordering::Acquire);
                    if prev <= 0 {
                        break;
                    }
                    if budget
                        .compare_exchange(prev, prev - 1, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        got_slot = true;
                        break;
                    }
                }
                if !got_slot {
                    break;
                }
            }
            consumed += 1;
        }
    } else {
        consumed = key.event_cache.len();
    }
    if consumed == 0 {
        return;
    }
    let drained: Vec<KeyNoteEvent> = key.event_cache.drain(..consumed).collect();
    for e in drained {
        key.data.send_event(e, control_data, cc_state, channel_sf, layers);
    }
}

impl Key {
    pub fn new(
        key: u8,
        shared_voice_counter: Arc<AtomicU64>,
        shared_group_id: Arc<AtomicU64>,
        options: ChannelInitOptions,
    ) -> Self {
        Key {
            data: KeyData::new(key, shared_voice_counter, shared_group_id, options),
            audio_cache: Vec::new(),
            event_cache: Vec::new(),
        }
    }
}

/// Options for initializing a new VoiceChannel.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(default)
)]
pub struct ChannelInitOptions {
    /// If set to true, the voices killed due to the voice limit will fade out.
    /// If set to false, they will be killed immediately, usually causing clicking
    /// but improving performance.
    ///
    /// Default: `false`
    pub fade_out_killing: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for ChannelInitOptions {
    fn default() -> Self {
        Self {
            fade_out_killing: false,
        }
    }
}

/// Represents a single MIDI channel within XSynth.
///
/// Keeps track and manages MIDI events and the active voices of a channel.
///
/// MIDI CC Support Chart:
/// - `CC0`: Bank Select
/// - `CC6`, `CC38`, `CC100`, `CC101`: RPN & NRPN
/// - `CC7`: Volume
/// - `CC8`: Balance
/// - `CC10`: Pan
/// - `CC11`: Expression
/// - `CC64`: Damper pedal
/// - `CC71`: Cutoff resonance
/// - `CC72`: Release time multiplier
/// - `CC73`: Attack time multiplier
/// - `CC74`: Cutoff frequency
/// - `CC120`: All sounds off
/// - `CC121`: Reset all controllers
/// - `CC123`: All notes off
pub struct VoiceChannel {
    key_voices: Vec<Key>,

    params: VoiceChannelParams,
    threadpool: Option<Arc<rayon::ThreadPool>>,

    stream_params: AudioStreamParams,

    /// The helper struct for keeping track of MIDI control event data
    control_event_data: ControlEventData,

    /// Processed control data, ready to feed to voices
    voice_control_data: VoiceControlData,

    /// MOVE FORK: per-channel raw MIDI CC array. Every `ControlEvent::Raw`
    /// stores its value here (in addition to whatever specialized
    /// handler runs). Voices will clone the Arc at spawn time to read
    /// live `_oncc` modulation under Ordering::Relaxed. Unused by voice
    /// generators today — Step 5 (SIMDVoiceOnccAmp) will consume it.
    /// ResetControl wipes back to zeros.
    cc_state: CcState,

    /// Effects
    cutoff: MultiChannelBiQuad,

    /// MOVE FORK / Phase 9 prototype: fundsp reverb on the channel's
    /// post-mix bus. `wet_send` is the dry/wet mix (0 = dry only,
    /// 1 = wet only). Default 0 so existing presets sound identical
    /// until something opts in. Updated via `SetChannelReverb`.
    reverb: Option<Box<dyn fundsp::prelude::AudioUnit + Send>>,
    reverb_wet: f32,
    /// MOVE FORK / Phase 9: hand-rolled stereo feedback delay. Only
    /// allocated when SetDelay enables it (preset has `<effect type=
    /// "delay">`). `delay_mix` 0..1 — zero skips processing entirely.
    delay: Option<crate::effects::StereoFeedbackDelay>,
    delay_mix: f32,
}

impl VoiceChannel {
    /// Initializes a new voice channel.
    ///
    /// - `options`: Channel configuration
    /// - `stream_params`: Parameters of the output audio
    /// - `threadpool`: The thread-pool that will be used to render the individual
    ///   keys' voices concurrently. If None is used, the voices will be
    ///   rendered on the same thread.
    pub fn new(
        options: ChannelInitOptions,
        stream_params: AudioStreamParams,
        threadpool: Option<Arc<rayon::ThreadPool>>,
    ) -> VoiceChannel {
        fn fill_key_array<T, F: Fn(u8) -> T>(func: F) -> Vec<T> {
            let mut vec = Vec::with_capacity(128);
            for i in 0..128 {
                vec.push(func(i));
            }
            vec
        }

        let params = VoiceChannelParams::new(stream_params);
        let shared_voice_counter = params.stats.voice_counter.clone();
        // MOVE FORK: shared group-id allocator for polyphony-cap support.
        // All keys' VoiceBuffers draw from this so group ids are globally
        // orderable across the channel (lower id = older note-on).
        let shared_group_id = Arc::new(AtomicU64::new(0));

        VoiceChannel {
            params,
            key_voices: fill_key_array(|i| {
                Key::new(i, shared_voice_counter.clone(), shared_group_id.clone(), options)
            }),

            threadpool,

            stream_params,

            control_event_data: ControlEventData::new_defaults(stream_params.sample_rate),
            voice_control_data: VoiceControlData::new_defaults(),
            cc_state: new_cc_state(),

            cutoff: MultiChannelBiQuad::new(
                stream_params.channels.count() as usize,
                FilterType::LowPass,
                stream_params.sample_rate as f32 / 2.0,
                stream_params.sample_rate as f32,
                None,
            ),
            reverb: None,
            reverb_wet: 0.0,
            delay: None,
            delay_mix: 0.0,
        }
    }

    /// MOVE FORK / Phase 9 prototype: install / replace a fundsp reverb
    /// on this channel. Call with `None` to remove. `set_sample_rate`
    /// is invoked on installation so the reverb runs at the output
    /// rate.
    pub fn set_reverb(&mut self, reverb: Option<Box<dyn fundsp::prelude::AudioUnit + Send>>) {
        if let Some(mut r) = reverb {
            r.set_sample_rate(self.stream_params.sample_rate as f64);
            self.reverb = Some(r);
        } else {
            self.reverb = None;
        }
    }

    /// MOVE FORK / Phase 9 prototype: set the dry/wet mix (0..1).
    pub fn set_reverb_wet(&mut self, wet: f32) {
        self.reverb_wet = wet.clamp(0.0, 1.0);
    }

    fn apply_channel_effects(&mut self, out: &mut [f32]) {
        let control = &mut self.control_event_data;

        match self.stream_params.channels {
            ChannelCount::Mono => {
                // Volume
                for sample in out.iter_mut() {
                    let vol = control.volume.get_next() * control.expression.get_next();
                    let vol = vol.powi(2);
                    *sample *= vol;
                }
            }
            ChannelCount::Stereo => {
                // Volume
                for sample in out.chunks_mut(2) {
                    let vol = control.volume.get_next() * control.expression.get_next();
                    let vol = vol.powi(2);
                    sample[0] *= vol;
                    sample[1] *= vol;
                }

                // Pan
                for sample in out.chunks_mut(2) {
                    let pan = control.pan.get_next();
                    sample[0] *= ((pan * std::f32::consts::PI / 2.0).cos()).min(1.0);
                    sample[1] *= ((pan * std::f32::consts::PI / 2.0).sin()).min(1.0);
                }
            }
        }

        // Cutoff
        if let Some(cutoff) = control.cutoff {
            self.cutoff
                .set_filter_type(FilterType::LowPass, cutoff, control.resonance);
            self.cutoff.process(out);
        }

        // MOVE FORK / Phase 9/10: DS `wetLevel` is documented as
        // "the volume of the [delay|reverb] signal" — additive send,
        // NOT a dry/wet crossfade. Confirmed by canonical DS delay
        // example where the wetLevel knob defaults to 1.0 (would
        // mute the sampler under crossfade). FX_MIX (chorus/phaser)
        // IS crossfade per DS convention; FX_WET_LEVEL is additive.
        //
        // Topology: `out = dry + send · wet`. Dry stays at unity;
        // the wet bus rides underneath at `send` (the author's
        // wetLevel). No engine-specific trim — author intent rules.
        if self.reverb_wet > 0.0 {
            if let Some(reverb) = self.reverb.as_mut() {
                let send = self.reverb_wet;
                if let ChannelCount::Stereo = self.stream_params.channels {
                    for sample in out.chunks_mut(2) {
                        let input = [sample[0], sample[1]];
                        let mut output = [0.0f32; 2];
                        reverb.tick(&input, &mut output);
                        sample[0] = sample[0] + output[0] * send;
                        sample[1] = sample[1] + output[1] * send;
                    }
                }
            }
        }

        if self.delay_mix > 0.0 {
            if let Some(delay) = self.delay.as_mut() {
                let send = self.delay_mix;
                if let ChannelCount::Stereo = self.stream_params.channels {
                    for sample in out.chunks_mut(2) {
                        let (wl, wr) = delay.process(sample[0], sample[1]);
                        sample[0] = sample[0] + wl * send;
                        sample[1] = sample[1] + wr * send;
                    }
                }
            }
        }
    }

    /// MOVE FORK: drop oldest note-groups so that polyphony stays at cap
    /// throughout the upcoming block — including after pending NoteOns
    /// drain. Counts NoteOns waiting in per-key event caches (bounded by
    /// the spawn-burst limit if set), and pre-drops `current + pending -
    /// cap` groups before the drain runs. Combined with `drop_group`'s
    /// instant-remove semantics, this gives a hard ceiling: voice count
    /// can never exceed `cap × voices_per_note` in any block, even during
    /// quantized chord bursts on top of pedal-held sustain.
    ///
    /// Drop priority: releasing groups first (their tails were fading
    /// anyway), then oldest groups regardless of state.
    fn enforce_polyphony_cap(&mut self) {
        let cap = match self.params.polyphony_cap {
            Some(c) => c,
            None => return,
        };
        let burst_limit = self.params.spawn_burst_limit.unwrap_or(usize::MAX);

        // Count pending NoteOns that will drain this block. Capped by
        // spawn-burst so we don't pre-drop for events that won't actually
        // arrive until next block.
        let mut pending_noteons = 0usize;
        for key in self.key_voices.iter() {
            for e in key.event_cache.iter() {
                if matches!(e, KeyNoteEvent::On(_)) {
                    pending_noteons += 1;
                    if pending_noteons >= burst_limit {
                        break;
                    }
                }
            }
            if pending_noteons >= burst_limit {
                break;
            }
        }

        // Collect every group across every key: (key_idx, group_id, all_releasing).
        let mut groups: Vec<(usize, u64, bool)> = Vec::new();
        let mut buf: Vec<(u64, bool)> = Vec::new();
        for (ki, key) in self.key_voices.iter().enumerate() {
            buf.clear();
            key.data.collect_groups(&mut buf);
            for &(gid, rel) in &buf {
                groups.push((ki, gid, rel));
            }
        }
        let current_polyphony = groups.len();
        let projected = current_polyphony + pending_noteons;
        if projected <= cap {
            return;
        }
        let drops_needed = projected - cap;

        // Sort: releasing-first (cheaper to drop, already curving down),
        // then by group id ascending (oldest first within each category).
        groups.sort_by(|a, b| b.2.cmp(&a.2).then(a.1.cmp(&b.1)));

        for &(ki, gid, _) in groups.iter().take(drops_needed) {
            self.key_voices[ki].data.drop_group(gid);
        }
    }

    fn push_key_events_and_render(&mut self, out: &mut [f32]) {
        let t_total = Instant::now();
        self.params.load_program();
        self.enforce_polyphony_cap();

        // MOVE FORK: shared spawn-burst budget for this block. When set,
        // worker threads CAS-decrement before processing each NoteOn;
        // any worker that can't claim a slot leaves the remaining events
        // (in their original order) in the key's event_cache for the
        // next block to pick up first. Drain stays in the parallel
        // section so the spawn-burst doesn't serialize spawn cost on the
        // audio thread — that single change alone added ~400 µs to the
        // burst block render time on Move.
        let spawn_budget: Option<Arc<std::sync::atomic::AtomicI64>> = self
            .params
            .spawn_burst_limit
            .map(|n| Arc::new(std::sync::atomic::AtomicI64::new(n as i64)));

        out.fill(0.0);
        let t_parallel = Instant::now();
        match self.threadpool.as_ref() {
            Some(pool) => {
                let len = out.len();
                let key_voices = &mut self.key_voices;
                let params = &self.params;
                let control_data = &self.voice_control_data;
                let cc_state = &self.cc_state;
                let spawn_budget = spawn_budget.as_ref();
                pool.install(|| {
                    key_voices.par_iter_mut().for_each(move |key| {
                        drain_events_with_budget(
                            key,
                            spawn_budget,
                            control_data,
                            cc_state,
                            &params.channel_sf,
                            params.layers,
                        );

                        prepapre_cache_vec(&mut key.audio_cache, len, 0.0);
                        key.data.render_to(&mut key.audio_cache);
                    });
                });
                let parallel_us = t_parallel.elapsed().as_micros() as u32;
                LAST_PARALLEL_US.store(parallel_us, Ordering::Relaxed);

                let t_sum = Instant::now();
                for key in self.key_voices.iter() {
                    sum_simd(&key.audio_cache, out);
                }
                LAST_SUM_US.store(t_sum.elapsed().as_micros() as u32, Ordering::Relaxed);
            }
            None => {
                for key in self.key_voices.iter_mut() {
                    drain_events_with_budget(
                        key,
                        spawn_budget.as_ref(),
                        &self.voice_control_data,
                        &self.cc_state,
                        &self.params.channel_sf,
                        self.params.layers,
                    );

                    key.data.render_to(out);
                }
                LAST_PARALLEL_US.store(t_parallel.elapsed().as_micros() as u32, Ordering::Relaxed);
                LAST_SUM_US.store(0, Ordering::Relaxed);
            }
        }

        let t_fx = Instant::now();
        self.apply_channel_effects(out);
        LAST_FX_US.store(t_fx.elapsed().as_micros() as u32, Ordering::Relaxed);
        LAST_TOTAL_US.store(t_total.elapsed().as_micros() as u32, Ordering::Relaxed);
    }

    fn propagate_voice_controls(&mut self) {
        for key in self.key_voices.iter_mut() {
            key.data.process_controls(&self.voice_control_data);
        }
    }

    fn kill_voices_in_exclusive_class(&mut self, class: u8) {
        for key in self.key_voices.iter_mut() {
            key.data.kill_by_exclusive_class(class);
        }
    }

    /// Sends a ChannelEvent to the channel.
    /// See the `ChannelEvent` documentation for more information.
    pub fn process_event(&mut self, event: ChannelEvent) {
        self.push_events_iter(std::iter::once(event));
    }

    /// Sends multiple ChannelEvent items to the channel as an iterator.
    pub fn push_events_iter<T: Iterator<Item = ChannelEvent>>(&mut self, iter: T) {
        for e in iter {
            match e {
                ChannelEvent::Audio(audio) => match audio {
                    ChannelAudioEvent::NoteOn { key, vel } => {
                        let classes: Vec<_> = self
                            .params
                            .channel_sf
                            .exclusive_classes_attack(key, vel)
                            .collect();
                        for class in classes {
                            self.kill_voices_in_exclusive_class(class);
                        }
                        if let Some(key) = self.key_voices.get_mut(key as usize) {
                            let ev = KeyNoteEvent::On(vel);
                            key.event_cache.push(ev);
                        }
                    }
                    ChannelAudioEvent::NoteOff { key } => {
                        if let Some(key) = self.key_voices.get_mut(key as usize) {
                            let ev = KeyNoteEvent::Off;
                            key.event_cache.push(ev);
                        }
                    }
                    ChannelAudioEvent::AllNotesOff => {
                        for key in self.key_voices.iter_mut() {
                            let ev = KeyNoteEvent::AllOff;
                            key.event_cache.push(ev);
                        }
                    }
                    ChannelAudioEvent::AllNotesKilled => {
                        for key in self.key_voices.iter_mut() {
                            let ev = KeyNoteEvent::AllKilled;
                            key.event_cache.push(ev);
                        }
                    }
                    ChannelAudioEvent::ResetControl => {
                        self.reset_control();
                    }
                    ChannelAudioEvent::Control(control) => {
                        self.process_control_event(control);
                    }
                    ChannelAudioEvent::ProgramChange(preset) => {
                        self.params.set_preset(preset);
                    }
                    ChannelAudioEvent::SystemReset => {
                        for key in self.key_voices.iter_mut() {
                            key.event_cache.clear();
                            key.event_cache.push(KeyNoteEvent::AllKilled);
                        }
                        self.reset_control();
                        self.reset_program();
                    }
                },
                ChannelEvent::Config(config) => {
                    // MOVE FORK / Phase 9 prototype: reverb config lives
                    // on VoiceChannel (post-mix state), not on
                    // VoiceChannelParams (voice-spawn config). Intercept
                    // here.
                    match config {
                        ChannelConfigEvent::SetReverb(params) => {
                            if let Some((room, time, damp)) = params {
                                let mut unit: Box<dyn fundsp::prelude::AudioUnit + Send> =
                                    Box::new(fundsp::hacker32::reverb_stereo(
                                        room, time, damp,
                                    ));
                                unit.set_sample_rate(
                                    self.stream_params.sample_rate as f64,
                                );
                                self.reverb = Some(unit);
                            } else {
                                self.reverb = None;
                            }
                        }
                        ChannelConfigEvent::SetReverbWet(wet) => {
                            self.reverb_wet = wet.clamp(0.0, 1.0);
                        }
                        ChannelConfigEvent::SetDelay(params) => {
                            if let Some((time, fb)) = params {
                                let mut d = crate::effects::StereoFeedbackDelay::new(
                                    self.stream_params.sample_rate as f32,
                                );
                                d.set_time(time);
                                d.set_feedback(fb);
                                self.delay = Some(d);
                            } else {
                                self.delay = None;
                            }
                        }
                        ChannelConfigEvent::SetDelayTime(t) => {
                            if let Some(d) = self.delay.as_mut() {
                                d.set_time(t);
                            }
                        }
                        ChannelConfigEvent::SetDelayFeedback(fb) => {
                            if let Some(d) = self.delay.as_mut() {
                                d.set_feedback(fb);
                            }
                        }
                        ChannelConfigEvent::SetDelayMix(m) => {
                            self.delay_mix = m.clamp(0.0, 1.0);
                        }
                        other => self.params.process_config_event(other),
                    }
                }
            }
        }
    }

    /// Returns a reader for the VoiceChannel statistics.
    /// See the `VoiceChannelStatsReader` documentation for more information.
    pub fn get_channel_stats(&self) -> VoiceChannelStatsReader {
        let stats = self.params.stats.clone();
        VoiceChannelStatsReader::new(stats)
    }

    /// MOVE FORK: cheap "does this channel have anything to render?" check.
    /// Used by ChannelGroup::render_to to skip empty channels — at 16 MIDI
    /// channels and one active SFZ instance, skipping the 15 idle channels
    /// shaves ~800 µs per render block (buffer fills, 128-key parallel
    /// iter, apply_channel_effects sweep). Must be called AFTER
    /// flush_events so events queued at the group level have been pushed
    /// to per-key caches.
    pub fn has_work(&self) -> bool {
        let voice_count = self
            .params
            .stats
            .voice_counter
            .load(std::sync::atomic::Ordering::Relaxed);
        if voice_count > 0 {
            return true;
        }
        for key in self.key_voices.iter() {
            if !key.event_cache.is_empty() {
                return true;
            }
        }
        // MOVE FORK / Phase 9 prototype: keep rendering while the
        // post-mix reverb is wet so its tail isn't truncated when the
        // last voice ends. Costs ~800 µs/block on idle channels, but
        // only when the user opts into reverb (default wet = 0).
        if self.reverb_wet > 0.0 && self.reverb.is_some() {
            return true;
        }
        // Same for delay — drain the feedback tail after the last note.
        if self.delay_mix > 0.0 && self.delay.is_some() {
            return true;
        }
        false
    }

    /// MOVE FORK: register a deferred-drop sink (see channel_sf::SoundfontDropSink).
    pub fn set_soundfont_drop_sink(&mut self, sink: SoundfontDropSink) {
        self.params.channel_sf.set_drop_sink(sink);
    }
}

impl AudioPipe for VoiceChannel {
    fn stream_params(&self) -> &AudioStreamParams {
        &self.params.constant.stream_params
    }

    fn read_samples_unchecked(&mut self, out: &mut [f32]) {
        self.push_key_events_and_render(out);
    }
}
