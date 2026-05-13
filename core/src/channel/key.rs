use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use super::{
    channel_sf::ChannelSoundfont, event::KeyNoteEvent, voice_buffer::VoiceBuffer,
    ChannelInitOptions, VoiceControlData,
};
use crate::voice::CcState;

/// MOVE FORK: append one diagnostic line to xsynth_debug.log per
/// NoteOn/NoteOff/AllOff. Notes are sparse (<100/sec on heavy play)
/// so file I/O on the audio thread is tolerable here. Used to hunt
/// the "random infinite release" bug — pairs of On/Off lines plus
/// state counts let us see when a NoteOff fails to release the just-
/// played group.
fn log_event(args: std::fmt::Arguments<'_>) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/data/UserData/schwung/tmp/xsynth_debug.log")
    {
        let _ = writeln!(f, "{}", args);
    }
}

pub struct KeyData {
    key: u8,
    voices: VoiceBuffer,
    last_voice_count: usize,
    shared_voice_counter: Arc<AtomicU64>,
}

impl KeyData {
    pub fn new(
        key: u8,
        shared_voice_counter: Arc<AtomicU64>,
        shared_group_id: Arc<AtomicU64>,
        options: ChannelInitOptions,
    ) -> KeyData {
        KeyData {
            key,
            voices: VoiceBuffer::new(options, shared_group_id),
            last_voice_count: 0,
            shared_voice_counter,
        }
    }

    pub fn send_event(
        &mut self,
        event: KeyNoteEvent,
        control: &VoiceControlData,
        cc_state: &CcState,
        channel_sf: &ChannelSoundfont,
        max_layers: Option<usize>,
    ) {
        match event {
            KeyNoteEvent::On(vel) => {
                let before = self.voices.voice_count();
                let voices = channel_sf.spawn_voices_attack(control, cc_state, self.key, vel);
                self.voices.push_voices(voices, max_layers);
                let (total, releasing, killed) = self.voices.voice_state_counts();
                log_event(format_args!(
                    "[ev] On  key={:>3} vel={:>3} spawned={} total={} releasing={} killed={}",
                    self.key, vel, total - before, total, releasing, killed,
                ));
            }
            KeyNoteEvent::Off => {
                let before_total = self.voices.voice_count();
                let vel = self.voices.release_next_voice();
                let mut spawned_rt = 0usize;
                if let Some(vel) = vel {
                    let before_rt = self.voices.voice_count();
                    let voices = channel_sf.spawn_voices_release(control, cc_state, self.key, vel);
                    // MOVE FORK: push release-trigger voices with the
                    // is_releasing flag pre-set, so a SUBSEQUENT NoteOff
                    // on this key skips them and properly releases the
                    // next attack voice group instead. Without this,
                    // we observed key=40 random infinite-release after
                    // a rapid On/Off cycle that left an RT voice at
                    // front of the buffer.
                    self.voices.push_release_trigger_voices(voices, max_layers);
                    spawned_rt = self.voices.voice_count() - before_rt;
                }
                let (total, releasing, killed) = self.voices.voice_state_counts();
                log_event(format_args!(
                    "[ev] Off key={:>3} released_vel={:?} rt_spawned={} before_total={} total={} releasing={} killed={}",
                    self.key, vel, spawned_rt, before_total, total, releasing, killed,
                ));
            }
            KeyNoteEvent::AllOff => {
                // MOVE FORK: snapshot non-releasing groups in one pass
                // and spawn release-trigger voices for each. The
                // previous `while let Some(vel) = release_next_voice()`
                // pattern infinite-looped on presets with release-
                // trigger regions (e.g. WörliTzer): each spawned
                // release-trigger voice is itself non-releasing, so
                // the next loop iteration would release it and spawn
                // MORE release-trigger voices, forever.
                let release_vels = self.voices.release_all_groups_snapshot();
                let n = release_vels.len();
                for vel in release_vels {
                    let voices = channel_sf.spawn_voices_release(control, cc_state, self.key, vel);
                    // MOVE FORK: same reason as the NoteOff path — RT
                    // voices must be pre-flagged as releasing.
                    self.voices.push_release_trigger_voices(voices, max_layers);
                }
                if n > 0 {
                    let (total, releasing, killed) = self.voices.voice_state_counts();
                    log_event(format_args!(
                        "[ev] AllOff key={:>3} groups_released={} total={} releasing={} killed={}",
                        self.key, n, total, releasing, killed,
                    ));
                }
            }
            KeyNoteEvent::AllKilled => {
                self.voices.kill_all_voices();
            }
        }
    }

    pub fn process_controls(&mut self, control: &VoiceControlData) {
        for voice in &mut self.voices.iter_voices_mut() {
            voice.process_controls(control);
        }
    }

    pub fn render_to(&mut self, out: &mut [f32]) {
        if self.has_voices() {
            for voice in &mut self.voices.iter_voices_mut() {
                voice.render_to(out);
            }
            self.voices.remove_ended_voices();
        }

        let voice_count = self.voices.voice_count();
        let change = voice_count as i64 - self.last_voice_count as i64;
        if change < 0 {
            self.shared_voice_counter
                .fetch_sub((-change) as u64, Ordering::SeqCst);
        } else {
            self.shared_voice_counter
                .fetch_add(change as u64, Ordering::SeqCst);
        }
        self.last_voice_count = voice_count;
    }

    pub fn has_voices(&self) -> bool {
        self.voices.has_voices()
    }

    pub fn set_damper(&mut self, damper: bool) {
        self.voices.set_damper(damper);
    }

    pub fn kill_by_exclusive_class(&mut self, class: u8) {
        self.voices.kill_by_exclusive_class(class);
    }

    /// MOVE FORK: passthrough for polyphony-cap enforcement.
    pub fn collect_groups(&self, out: &mut Vec<(u64, bool)>) {
        self.voices.collect_groups(out);
    }

    /// MOVE FORK: drop a whole group; returns voices removed. The next
    /// `render_to` will reconcile `shared_voice_counter`.
    pub fn drop_group(&mut self, group_id: u64) -> usize {
        self.voices.drop_group(group_id)
    }
}
