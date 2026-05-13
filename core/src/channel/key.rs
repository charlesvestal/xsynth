use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use super::{
    channel_sf::ChannelSoundfont, event::KeyNoteEvent, voice_buffer::VoiceBuffer,
    ChannelInitOptions, VoiceControlData,
};

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
        channel_sf: &ChannelSoundfont,
        max_layers: Option<usize>,
    ) {
        match event {
            KeyNoteEvent::On(vel) => {
                let voices = channel_sf.spawn_voices_attack(control, self.key, vel);
                self.voices.push_voices(voices, max_layers);
            }
            KeyNoteEvent::Off => {
                let vel = self.voices.release_next_voice();
                if let Some(vel) = vel {
                    let voices = channel_sf.spawn_voices_release(control, self.key, vel);
                    self.voices.push_voices(voices, max_layers);
                }
            }
            KeyNoteEvent::AllOff => {
                while let Some(vel) = self.voices.release_next_voice() {
                    let voices = channel_sf.spawn_voices_release(control, self.key, vel);
                    self.voices.push_voices(voices, max_layers);
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
