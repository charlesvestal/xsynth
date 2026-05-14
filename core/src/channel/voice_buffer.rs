use super::ChannelInitOptions;
use crate::voice::{ReleaseType, Voice};
use std::{
    collections::VecDeque,
    fmt::Debug,
    ops::{Deref, DerefMut},
    sync::{atomic::{AtomicU64, Ordering}, Arc},
};

struct GroupVoice {
    /// MOVE FORK: group id is globally unique across all keys in a
    /// channel (allocated from a shared atomic counter). Lower id =
    /// older note-on. Used by the polyphony-cap enforcer to pick the
    /// oldest releasing group across the channel.
    pub id: u64,
    pub voice: Box<dyn Voice>,
}

impl Deref for GroupVoice {
    type Target = Box<dyn Voice>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.voice
    }
}

impl DerefMut for GroupVoice {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Box<dyn Voice> {
        &mut self.voice
    }
}

impl Debug for GroupVoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("")
            .field(&self.id)
            .field(&self.voice.velocity())
            .field(&self.voice.is_killed())
            .finish()
    }
}

pub struct VoiceBuffer {
    options: ChannelInitOptions,
    /// MOVE FORK: channel-shared id allocator. All VoiceBuffers in the
    /// same VoiceChannel share this Arc so group ids are globally
    /// orderable. Lower = older.
    id_counter: Arc<AtomicU64>,
    buffer: VecDeque<GroupVoice>,
    damper_held: bool,
    held_by_damper: Vec<u64>,
}

impl VoiceBuffer {
    pub fn new(options: ChannelInitOptions, id_counter: Arc<AtomicU64>) -> Self {
        VoiceBuffer {
            options,
            id_counter,
            buffer: VecDeque::new(),
            damper_held: false,
            held_by_damper: Vec::new(),
        }
    }

    fn get_id(&mut self) -> u64 {
        self.id_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Pops the quietest voice group. Multiple voices can be part of the same group
    /// based on their ID (e.g. a note and a hammer playing at the same time for a note on event)
    fn pop_quietest_voice_group(&mut self, ignored_id: u64) {
        if self.buffer.is_empty() {
            return;
        }

        let mut quietest = u8::MAX;
        let mut quietest_index = 0;
        let mut quietest_id = 0u64;
        let mut count = 0;
        for i in 0..self.buffer.len() {
            let voice = &self.buffer[i];
            if voice.id == ignored_id || voice.is_killed() {
                continue;
            }
            let vel = voice.velocity();
            if quietest_id == voice.id {
                count += 1;
            } else if vel < quietest || i == 0 {
                quietest = vel;
                quietest_index = i;
                quietest_id = voice.id;
                count = 1;
            }
        }

        if count > 0 {
            if self.options.fade_out_killing {
                for i in quietest_index..(quietest_index + count) {
                    self.kill_voice_fade_out(i);
                }
            } else {
                self.buffer.drain(quietest_index..(quietest_index + count));
            }

            if let Some(index) = self.held_by_damper.iter().position(|&x| x == quietest_id) {
                self.held_by_damper.remove(index);
            }
        }
    }

    fn kill_voice_fade_out(&mut self, index: usize) {
        self.buffer[index]
            .deref_mut()
            .signal_release(ReleaseType::Kill);
    }

    pub fn kill_all_voices(&mut self) {
        if self.options.fade_out_killing {
            for i in 0..self.buffer.len() {
                self.kill_voice_fade_out(i);
            }
        } else {
            self.buffer.clear();
        }
    }

    pub fn kill_by_exclusive_class(&mut self, class: u8) {
        for voice in &mut self.buffer {
            if voice.exclusive_class() == Some(class) {
                voice.signal_release(ReleaseType::Kill);
            }
        }
    }

    fn get_active_count(&mut self) -> usize {
        let mut active = 0;
        for i in 0..self.buffer.len() {
            if !self.buffer[i].deref().is_killed() {
                active += 1;
            }
        }
        active
    }

    /// MOVE FORK: like `push_voices`, but marks every voice as
    /// "release-trigger" so it's invisible to `release_next_voice`
    /// and `release_all_groups_snapshot`. Use for the voices spawned
    /// by `spawn_voices_release` (i.e. `trigger=release` samples).
    ///
    /// Without this, a NoteOff that spawned a release-trigger voice
    /// would itself become a candidate for the next NoteOff's release:
    /// release_next_voice would find the RT voice at the front of the
    /// buffer (non-releasing, non-killed) and "release" it, taking
    /// the slot from the actually-just-played voice — which would
    /// then play its sample to the end, audibly ignoring the user's
    /// key-up. (Diagnosed on WörliTzer key 40: rapid Off/On/Off
    /// sequence repeatedly mis-released the RT voice instead of the
    /// new attack voices.)
    pub fn push_release_trigger_voices(
        &mut self,
        voices: impl Iterator<Item = Box<dyn Voice>>,
        max_voices: Option<usize>,
    ) {
        let mut len = 0;
        let id = self.get_id();
        for mut voice in voices {
            voice.mark_as_release_trigger();
            self.buffer.push_back(GroupVoice { id, voice });
            len += 1;
        }

        // Same polyphony-cap path as push_voices — release-trigger
        // voices still count against per-key layer cap (relevant when
        // a layered preset has many release-trigger regions).
        if let Some(max_voices) = max_voices {
            if len > max_voices {
                self.pop_quietest_voice_group(id);
            } else if self.options.fade_out_killing {
                while self.get_active_count() > max_voices {
                    self.pop_quietest_voice_group(id);
                }
            } else {
                while self.buffer.len() > max_voices {
                    self.pop_quietest_voice_group(id);
                }
            }
        }
    }

    /// Pushes a new set of voices for a single note on event. Multiple voices can be part of the same group
    /// based on their ID (e.g. a note and a hammer playing at the same time for a note on event)
    pub fn push_voices(
        &mut self,
        voices: impl Iterator<Item = Box<dyn Voice>>,
        max_voices: Option<usize>,
    ) {
        let mut len = 0;

        let id = self.get_id();
        for voice in voices {
            self.buffer.push_back(GroupVoice { id, voice });
            len += 1;
        }

        if let Some(max_voices) = max_voices {
            if len > max_voices {
                self.pop_quietest_voice_group(id);
            } else if self.options.fade_out_killing {
                while self.get_active_count() > max_voices {
                    self.pop_quietest_voice_group(id);
                }
            } else {
                while self.buffer.len() > max_voices {
                    self.pop_quietest_voice_group(id);
                }
            }
        }
    }

    /// MOVE FORK: snapshot every currently-non-releasing voice group,
    /// mark them all as releasing in one pass, return their velocities
    /// for the caller to spawn `trigger=release` samples per group.
    ///
    /// Avoids the infinite-loop pattern in the caller:
    ///     while let Some(vel) = self.release_next_voice() {
    ///         let voices = channel_sf.spawn_voices_release(...);
    ///         self.push_voices(voices, ...);
    ///     }
    /// Those newly-pushed release-trigger voices are themselves
    /// non-releasing, so the loop's next `release_next_voice` finds and
    /// releases them, which spawns MORE release-trigger voices, ad
    /// infinitum on presets like WörliTzer that have release-trigger
    /// groups. The snapshot only iterates the groups that existed at
    /// entry; voices spawned by spawn_voices_release are left to play
    /// their tails without being re-released.
    ///
    /// damper_held: skipped (matches release_next_voice's else branch —
    /// release deferred until damper lifts).
    pub fn release_all_groups_snapshot(&mut self) -> Vec<u8> {
        if self.damper_held {
            // Mirror the per-call damper-held path: add each non-releasing,
            // not-already-tracked group's id to held_by_damper, no vels.
            // is_killed voices are skipped — they're already going away.
            for voice in self.buffer.iter_mut() {
                if voice.is_releasing() || voice.is_killed() {
                    continue;
                }
                if self.held_by_damper.contains(&voice.id) {
                    continue;
                }
                self.held_by_damper.push(voice.id);
            }
            return Vec::new();
        }
        let mut velocities = Vec::new();
        let mut last_id: Option<u64> = None;
        for voice in self.buffer.iter_mut() {
            if voice.is_releasing() || voice.is_killed() {
                continue;
            }
            if last_id != Some(voice.id) {
                velocities.push(voice.velocity());
                last_id = Some(voice.id);
            }
            voice.signal_release(ReleaseType::Standard);
        }
        velocities
    }

    /// Releases the next voice, and all subsequent voices that have the same ID.
    pub fn release_next_voice(&mut self) -> Option<u8> {
        if !self.damper_held {
            let mut id: Option<u64> = None;
            let mut vel = None;

            // Find the first non releasing voice, get its id and release all voices with that id.
            // MOVE FORK: ALSO skip is_killed() voices — drop_group (polyphony-cap
            // enforce) sets killed=true but leaves releasing=false. Without
            // this skip, a NoteOff on key K would "release" an old killed
            // group at the front of the buffer (a no-op since the voice is
            // already fading out via Kill) and the ACTUAL just-played voice
            // at K would never receive its release. Symptom: random notes
            // play their sample fully, ignoring key-up. Same fix applies to
            // release_all_groups_snapshot above.
            for voice in self.buffer.iter_mut() {
                if voice.is_releasing() || voice.is_killed() {
                    continue;
                }

                if id.is_none() {
                    id = Some(voice.id);
                    vel = Some(voice.velocity())
                }

                if id != Some(voice.id) {
                    break;
                }

                voice.signal_release(ReleaseType::Standard);
            }

            vel
        } else {
            // Find the first non releasing voice which also isn't being held in the release buffer, and add it to the release buffer
            for voice in self.buffer.iter_mut() {
                if voice.is_releasing() {
                    continue;
                }

                if self.held_by_damper.contains(&voice.id) {
                    continue;
                }

                self.held_by_damper.push(voice.id);
                break;
            }

            None
        }
    }

    pub fn remove_ended_voices(&mut self) {
        let mut i = 0;
        while i < self.buffer.len() {
            if self.buffer[i].ended() {
                self.buffer.remove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn iter_voices_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn Voice>> {
        self.buffer.iter_mut().map(|group| &mut group.voice)
    }

    pub fn has_voices(&self) -> bool {
        !self.buffer.is_empty()
    }

    pub fn voice_count(&self) -> usize {
        self.buffer.len()
    }

    pub fn set_damper(&mut self, damper: bool) {
        if self.damper_held && !damper {
            // Release all voices that are held by the damper
            for voice in self.buffer.iter_mut() {
                if self.held_by_damper.contains(&voice.id) {
                    voice.signal_release(ReleaseType::Standard);
                }
            }
            self.held_by_damper.clear();
        }
        self.damper_held = damper;
    }

    /// MOVE FORK: summarize the groups in this buffer for the polyphony-
    /// cap enforcer. Appends `(group_id, all_releasing)` per distinct id.
    /// `all_releasing` is true iff every voice in the group is either
    /// already releasing or already killed — those groups are the cheapest
    /// to drop (they were fading to silence anyway).
    pub fn collect_groups(&self, out: &mut Vec<(u64, bool)>) {
        let mut current: Option<(u64, bool)> = None;
        for v in &self.buffer {
            let releasing = v.is_releasing() || v.is_killed();
            match current {
                Some((cur_id, cur_rel)) if cur_id == v.id => {
                    current = Some((cur_id, cur_rel && releasing));
                }
                _ => {
                    if let Some(prev) = current {
                        out.push(prev);
                    }
                    current = Some((v.id, releasing));
                }
            }
        }
        if let Some(prev) = current {
            out.push(prev);
        }
    }

    /// MOVE FORK: drop a group via fast envelope-controlled fade-kill.
    /// Tried instant-remove for hard cap, but the click on pedal-held
    /// (full-amplitude) voices was too audible. Fade-kill is smoother;
    /// the tradeoff is that voices keep rendering for ~1 ms after the
    /// drop, so polyphony temporarily exceeds the cap by drops_needed
    /// voices during burst blocks. That's bounded by the burst limit
    /// (default 3 NoteOn/block).
    pub fn drop_group(&mut self, group_id: u64) -> usize {
        let mut count = 0;
        for v in self.buffer.iter_mut() {
            if v.id == group_id && !v.is_killed() {
                v.signal_release(ReleaseType::Kill);
                count += 1;
            }
        }
        self.held_by_damper.retain(|&id| id != group_id);
        count
    }
}
