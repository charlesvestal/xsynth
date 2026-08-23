use std::{marker::PhantomData, ops::Mul, sync::Arc};

use simdeez::Simd;

use crate::{
    effects::BiQuadFilter,
    voice::{
        SIMDMonoVoiceCutoffLive, SIMDSample, SIMDSampleGrabber, SIMDSampleMono,
        SIMDVoiceGenerator,
    },
    AudioStreamParams,
};
use crate::{
    voice::VoiceControlData,
    voice::{
        BufferSamplers, CcState, EnvelopeParameters, SIMDConstant, SIMDLinearSampleGrabber, SIMDMonoVoice,
        SIMDMonoVoiceSampler, SIMDNearestSampleGrabber, SIMDVoiceControl, SIMDVoiceEnvelope,
        SIMDVoiceLfoAmp, SIMDVoiceOnccAmp, SampleReader, SampleReaderLoop, SampleReaderLoopSustain,
        SampleReaderNoLoop, Voice, VoiceBase, VoiceCombineSIMD,
    },
};

use xsynth_soundfonts::LoopMode;

use crate::soundfont::{Interpolator, LoopParams, SampleSource, SampleVoiceSpawnerParams, VoiceSpawner};
#[allow(unused_imports)]
use crate::soundfont::SampleStorage;

pub struct MonoSampledVoiceSpawner<S: 'static + Simd + Send + Sync> {
    speed_mult: f32,
    filter: Option<BiQuadFilter>,
    /// MOVE FORK: filter params kept for live cutoff modulation.
    filter_type: xsynth_soundfonts::FilterType,
    base_cutoff: f32,
    base_resonance_db: f32,
    loop_params: LoopParams,
    amp: f32,
    volume_envelope_params: Arc<EnvelopeParameters>,
    /// MOVE FORK / 2026-05-19: live ampeg recompute path — see stereo.rs
    /// for the full explanation. Mono spawners get the same plumbing
    /// so SFZ files that author ampeg_*_oncc on a mono region work too.
    envelope_descriptor: crate::voice::EnvelopeDescriptor,
    envelope_options: crate::soundfont::EnvelopeOptions,
    ampeg_attack_oncc:  Arc<[(u8, f32, f32)]>,
    ampeg_decay_oncc:   Arc<[(u8, f32, f32)]>,
    ampeg_sustain_oncc: Arc<[(u8, f32, f32)]>,
    ampeg_release_oncc: Arc<[(u8, f32, f32)]>,
    sample_source: SampleSource,
    interpolator: Interpolator,
    exclusive_class: Option<u8>,
    vel: u8,
    stream_params: AudioStreamParams,
    /// MOVE FORK: see stereo.rs equivalent.
    volume_oncc: Arc<[(u8, f32)]>,
    /// MOVE FORK: live cutoff/resonance oncc bindings.
    cutoff_oncc: Arc<[(u8, f32)]>,
    resonance_oncc: Arc<[(u8, f32)]>,
    volume_curvecc: Arc<[(u8, u8)]>,
    cutoff_curvecc: Arc<[(u8, u8)]>,
    resonance_curvecc: Arc<[(u8, u8)]>,
    curves: Arc<std::collections::HashMap<u8, [f32; 128]>>,
    amp_lfo_freq: f32,
    amp_lfo_depth: f32,
    amp_lfo_freq_oncc: Arc<[(u8, f32)]>,
    amp_lfo_depth_oncc: Arc<[(u8, f32)]>,
    fil_lfo_freq: f32,
    fil_lfo_depth: f32,
    fil_lfo_freq_oncc: Arc<[(u8, f32)]>,
    fil_lfo_depth_oncc: Arc<[(u8, f32)]>,
    _s: PhantomData<S>,
}

impl<S: Simd + Send + Sync> MonoSampledVoiceSpawner<S> {
    pub fn new(
        params: &SampleVoiceSpawnerParams,
        vel: u8,
        stream_params: AudioStreamParams,
    ) -> Self {
        let amp = params.volume;

        let filter = params.cutoff.map(|cutoff| {
            BiQuadFilter::new(
                params.filter_type,
                cutoff,
                stream_params.sample_rate as f32,
                Some(params.resonance),
            )
        });

        Self {
            speed_mult: params.speed_mult,
            filter,
            filter_type: params.filter_type,
            base_cutoff: params.cutoff.unwrap_or(22000.0),
            base_resonance_db: params.base_resonance_db,
            loop_params: params.loop_params.clone(),
            amp,
            volume_envelope_params: params.envelope.clone(),
            envelope_descriptor: params.envelope_descriptor,
            envelope_options: params.envelope_options,
            ampeg_attack_oncc:  params.ampeg_attack_oncc.clone(),
            ampeg_decay_oncc:   params.ampeg_decay_oncc.clone(),
            ampeg_sustain_oncc: params.ampeg_sustain_oncc.clone(),
            ampeg_release_oncc: params.ampeg_release_oncc.clone(),
            sample_source: params.sample.clone(),
            interpolator: params.interpolator,
            exclusive_class: params.exclusive_class,
            vel,
            stream_params,
            volume_oncc: params.volume_oncc.clone(),
            cutoff_oncc: params.cutoff_oncc.clone(),
            resonance_oncc: params.resonance_oncc.clone(),
            volume_curvecc: params.volume_curvecc.clone(),
            cutoff_curvecc: params.cutoff_curvecc.clone(),
            resonance_curvecc: params.resonance_curvecc.clone(),
            curves: params.curves.clone(),
            amp_lfo_freq: params.amp_lfo_freq,
            amp_lfo_depth: params.amp_lfo_depth,
            amp_lfo_freq_oncc: params.amp_lfo_freq_oncc.clone(),
            amp_lfo_depth_oncc: params.amp_lfo_depth_oncc.clone(),
            fil_lfo_freq: params.fil_lfo_freq,
            fil_lfo_depth: params.fil_lfo_depth,
            fil_lfo_freq_oncc: params.fil_lfo_freq_oncc.clone(),
            fil_lfo_depth_oncc: params.fil_lfo_depth_oncc.clone(),
            _s: PhantomData,
        }
    }

    /// MOVE FORK / 2026-05-16: build a single-channel BufferSamplers from
    /// the SampleSource. Mono spawner always uses channel 0.
    fn build_buffer_sampler(&self) -> BufferSamplers {
        match &self.sample_source {
            SampleSource::Heap(arr) => BufferSamplers::new_f32(arr[0].clone()),
            #[cfg(unix)]
            SampleSource::Streamed { source, io_pool, head, head_start } => {
                let empty: Arc<[i16]> = Arc::from(Vec::new().into_boxed_slice());
                let head_0 = head.get(0).cloned().unwrap_or(empty);
                let stream = io_pool.register(
                    source.clone(), 0, head_0, *head_start as usize,
                );
                BufferSamplers::streamed(stream)
            }
        }
    }

    fn begin_voice(&self, control: &VoiceControlData, cc_state: &CcState) -> Box<dyn Voice> {
        let bs = self.build_buffer_sampler();
        match self.loop_params.mode {
            LoopMode::LoopContinuous => {
                let r = SampleReaderLoop::new(bs, self.loop_params.clone());
                self.dispatch_interpolator(control, cc_state, r)
            }
            LoopMode::LoopSustain => {
                let r = SampleReaderLoopSustain::new(bs, self.loop_params.clone());
                self.dispatch_interpolator(control, cc_state, r)
            }
            LoopMode::NoLoop | LoopMode::OneShot => {
                let r = SampleReaderNoLoop::new(bs, self.loop_params.clone());
                self.dispatch_interpolator(control, cc_state, r)
            }
        }
    }

    fn dispatch_interpolator<SR: 'static + SampleReader>(
        &self,
        control: &VoiceControlData,
        cc_state: &CcState,
        reader: SR,
    ) -> Box<dyn Voice> {
        match self.interpolator {
            Interpolator::Nearest => {
                let g = SIMDNearestSampleGrabber::new(reader);
                self.generate_sampler(control, cc_state, g)
            }
            Interpolator::Linear => {
                let g = SIMDLinearSampleGrabber::new(reader);
                self.generate_sampler(control, cc_state, g)
            }
        }
    }

    fn generate_sampler<SG: 'static + SIMDSampleGrabber<S>>(
        &self,
        control: &VoiceControlData,
        cc_state: &CcState,
        grabber: SG,
    ) -> Box<dyn Voice> {
        let pitch_fac = self.create_pitch_fac(control);
        let sampler = SIMDMonoVoiceSampler::new(grabber, pitch_fac);
        self.apply_voice_params(sampler, control, cc_state)
    }

    fn apply_velocity<Gen, Sample>(&self, gen: Gen) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleMono<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let amp = SIMDConstant::<S>::new(self.amp);
        let amp = VoiceCombineSIMD::mult(amp, gen);
        amp
    }

    fn create_pitch_fac(
        &self,
        control: &VoiceControlData,
    ) -> impl SIMDVoiceGenerator<S, SIMDSampleMono<S>> {
        let pitch_fac = SIMDConstant::<S>::new(self.speed_mult);
        let pitch_multiplier = SIMDVoiceControl::new(control, |vc| vc.voice_pitch_multiplier);
        let pitch_fac = VoiceCombineSIMD::mult(pitch_fac, pitch_multiplier);
        pitch_fac
    }

    fn apply_envelope<Gen, Sample>(
        &self,
        gen: Gen,
        control: &VoiceControlData,
        cc_state: &CcState,
    ) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleMono<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let has_oncc = !self.ampeg_attack_oncc.is_empty()
            || !self.ampeg_decay_oncc.is_empty()
            || !self.ampeg_sustain_oncc.is_empty()
            || !self.ampeg_release_oncc.is_empty();
        let base_params = if has_oncc {
            let mut desc = self.envelope_descriptor;
            let extra = |list: &[(u8, f32, f32)]| -> f32 {
                let mut acc = 0.0_f32;
                for &(cc, value, cc_init) in list.iter() {
                    if (cc as usize) < 128 {
                        let cc_norm = cc_state[cc as usize]
                            .load(std::sync::atomic::Ordering::Relaxed)
                            as f32
                            / 127.0;
                        acc += value * (cc_norm - cc_init);
                    }
                }
                acc
            };

            desc.attack          = (desc.attack          + extra(&self.ampeg_attack_oncc)).max(0.0);
            desc.decay           = (desc.decay           + extra(&self.ampeg_decay_oncc)).max(0.0);
            desc.sustain_percent = (desc.sustain_percent + extra(&self.ampeg_sustain_oncc)).clamp(0.0, 1.0);
            desc.release         = (desc.release         + extra(&self.ampeg_release_oncc)).max(0.0);
            desc.to_envelope_params(
                self.stream_params.sample_rate,
                self.envelope_options,
            )
        } else {
            *self.volume_envelope_params.clone()
        };

        let modified_params = SIMDVoiceEnvelope::<S>::get_modified_envelope(
            base_params,
            control.envelope,
            self.stream_params.sample_rate as f32,
        );

        let allow_release = self.loop_params.mode != LoopMode::OneShot;

        let volume_envelope = SIMDVoiceEnvelope::new(
            base_params,
            modified_params,
            allow_release,
            self.stream_params.sample_rate as f32,
        );

        let amp = VoiceCombineSIMD::mult(volume_envelope, gen);
        amp
    }

    fn convert_to_voice<Gen>(&self, gen: Gen) -> Box<dyn Voice>
    where
        Gen: 'static + SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    {
        let flattened = SIMDMonoVoice::new(gen);
        let base = VoiceBase::new(self.vel, self.exclusive_class(), flattened);

        Box::new(base)
    }

    fn apply_voice_params<Gen>(
        &self,
        gen: Gen,
        control: &VoiceControlData,
        cc_state: &CcState,
    ) -> Box<dyn Voice>
    where
        Gen: 'static + SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    {
        let gen = self.apply_velocity(gen);
        let gen = self.apply_volume_oncc(gen, cc_state);
        let gen = self.apply_amp_lfo(gen, cc_state);
        let gen = self.apply_envelope(gen, control, cc_state);

        self.apply_cutoff_effect(gen, cc_state)
    }

    /// MOVE FORK / Phase 11: amp LFO (tremolo).
    fn apply_amp_lfo<Gen, Sample>(&self, gen: Gen, cc_state: &CcState) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleMono<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let lfo = SIMDVoiceLfoAmp::<S>::new(
            cc_state.clone(),
            self.amp_lfo_freq, self.amp_lfo_depth,
            self.stream_params.sample_rate as f32,
            self.amp_lfo_freq_oncc.clone(),
            self.amp_lfo_depth_oncc.clone(),
        );
        VoiceCombineSIMD::mult(lfo, gen)
    }

    /// MOVE FORK: see stereo.rs apply_volume_oncc for design notes.
    fn apply_volume_oncc<Gen, Sample>(
        &self,
        gen: Gen,
        cc_state: &CcState,
    ) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleMono<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let oncc = SIMDVoiceOnccAmp::<S>::new(
            cc_state.clone(),
            self.volume_oncc.clone(),
            self.volume_curvecc.clone(),
            self.curves.clone(),
        );
        VoiceCombineSIMD::mult(oncc, gen)
    }

    fn apply_cutoff_effect(
        &self,
        gen: impl 'static + SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
        cc_state: &CcState,
    ) -> Box<dyn Voice> {
        if let Some(filter) = &self.filter {
            let gen = SIMDMonoVoiceCutoffLive::new(
                gen,
                filter,
                cc_state.clone(),
                self.cutoff_oncc.clone(),
                self.resonance_oncc.clone(),
                self.cutoff_curvecc.clone(),
                self.resonance_curvecc.clone(),
                self.curves.clone(),
                self.fil_lfo_freq,
                self.fil_lfo_depth,
                self.fil_lfo_freq_oncc.clone(),
                self.fil_lfo_depth_oncc.clone(),
                // Mono spawner doesn't currently surface fileg; SF2
                // doesn't use it and SFZ mono is rare for our presets.
                0.0, 0.0, 1.0, 0.0, 0.0, None,
                self.filter_type,
                self.stream_params.sample_rate as f32,
                self.base_cutoff,
                self.base_resonance_db,
            );
            self.convert_to_voice(gen)
        } else {
            self.convert_to_voice(gen)
        }
    }
}

impl<S: 'static + Sync + Send + Simd> VoiceSpawner for MonoSampledVoiceSpawner<S> {
    fn spawn_voice(&self, control: &VoiceControlData, cc_state: &CcState) -> Box<dyn Voice> {
        self.begin_voice(control, cc_state)
    }

    fn exclusive_class(&self) -> Option<u8> {
        self.exclusive_class
    }
}
