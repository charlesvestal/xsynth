use std::{marker::PhantomData, ops::Mul, sync::Arc};

use simdeez::Simd;

use crate::{
    effects::BiQuadFilter,
    voice::{
        BufferSampler, SIMDSample, SIMDSampleGrabber, SIMDSampleMono, SIMDSampleStereo,
        SIMDStereoVoiceCutoff, SIMDVoiceGenerator,
    },
    AudioStreamParams,
};
use crate::{
    voice::VoiceControlData,
    voice::{
        BufferSamplers, CcState, EnvelopeParameters, SIMDConstant, SIMDConstantStereo,
        SIMDLinearSampleGrabber, SIMDNearestSampleGrabber, SIMDStereoVoice, SIMDStereoVoiceSampler,
        SIMDVoiceControl, SIMDVoiceEnvelope, SIMDVoiceOnccAmp, SampleReader, SampleReaderLoop,
        SampleReaderLoopSustain, SampleReaderNoLoop, Voice, VoiceBase, VoiceCombineSIMD,
    },
};

use xsynth_soundfonts::LoopMode;

use crate::soundfont::{Interpolator, LoopParams, SampleStorage, SampleVoiceSpawnerParams, VoiceSpawner};

pub struct StereoSampledVoiceSpawner<S: 'static + Simd + Send + Sync> {
    speed_mult: f32,
    filter: Option<BiQuadFilter>,
    loop_params: LoopParams,
    amp: f32,
    pan: f32,
    volume_envelope_params: Arc<EnvelopeParameters>,
    samples: Arc<[Arc<SampleStorage>]>,
    interpolator: Interpolator,
    exclusive_class: Option<u8>,
    vel: u8,
    stream_params: AudioStreamParams,
    /// MOVE FORK: live `volume_oncc` bindings — when non-empty the voice
    /// gets a SIMDVoiceOnccAmp stage that polls the channel CC array.
    volume_oncc: Arc<[(u8, f32)]>,
    _s: PhantomData<S>,
}

impl<S: Simd + Send + Sync> StereoSampledVoiceSpawner<S> {
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
            loop_params: params.loop_params.clone(),
            amp,
            pan: params.pan,
            volume_envelope_params: params.envelope.clone(),
            samples: params.sample.clone(),
            interpolator: params.interpolator,
            exclusive_class: params.exclusive_class,
            vel,
            stream_params,
            volume_oncc: params.volume_oncc.clone(),
            _s: PhantomData,
        }
    }

    fn begin_voice(&self, control: &VoiceControlData, cc_state: &CcState) -> Box<dyn Voice> {
        // Currently there's only the f32 buffer samples, more could be added in the future.
        #[allow(clippy::redundant_closure)]
        self.make_sample_reader(control, cc_state, |s| BufferSamplers::new_f32(s))
    }

    fn make_sample_reader<BS: 'static + BufferSampler>(
        &self,
        control: &VoiceControlData,
        cc_state: &CcState,
        make_bs: impl Fn(Arc<SampleStorage>) -> BS,
    ) -> Box<dyn Voice> {
        match self.loop_params.mode {
            LoopMode::LoopContinuous => self.make_sample_grabber(control, cc_state, move |s| {
                SampleReaderLoop::new(make_bs(s), self.loop_params.clone())
            }),
            LoopMode::LoopSustain => self.make_sample_grabber(control, cc_state, move |s| {
                SampleReaderLoopSustain::new(make_bs(s), self.loop_params.clone())
            }),
            LoopMode::NoLoop | LoopMode::OneShot => self.make_sample_grabber(control, cc_state, move |s| {
                SampleReaderNoLoop::new(make_bs(s), self.loop_params.clone())
            }),
        }
    }

    fn make_sample_grabber<SR: 'static + SampleReader>(
        &self,
        control: &VoiceControlData,
        cc_state: &CcState,
        make_bs: impl Fn(Arc<SampleStorage>) -> SR,
    ) -> Box<dyn Voice> {
        match self.interpolator {
            Interpolator::Nearest => {
                self.generate_sampler(control, cc_state, |s| SIMDNearestSampleGrabber::new(make_bs(s)))
            }
            Interpolator::Linear => {
                self.generate_sampler(control, cc_state, |s| SIMDLinearSampleGrabber::new(make_bs(s)))
            }
        }
    }

    fn generate_sampler<SG: 'static + SIMDSampleGrabber<S>>(
        &self,
        control: &VoiceControlData,
        cc_state: &CcState,
        make_sampler: impl Fn(Arc<SampleStorage>) -> SG,
    ) -> Box<dyn Voice> {
        let left = make_sampler(self.samples[0].clone());
        let right = make_sampler(self.samples[1].clone());

        let pitch_fac = self.create_pitch_fac(control);

        let sampler = SIMDStereoVoiceSampler::new(left, right, pitch_fac);
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

    fn apply_pan<Gen, Sample>(&self, gen: Gen) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleStereo<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let pan = self.pan * std::f32::consts::PI / 2.0;
        let leftg = (pan.cos() * 1.42).min(1.0);
        let rightg = (pan.sin() * 1.42).min(1.0);

        let gains = SIMDConstantStereo::<S>::new(leftg, rightg);

        let panned = VoiceCombineSIMD::mult(gains, gen);
        panned
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
    ) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleMono<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let modified_params = SIMDVoiceEnvelope::<S>::get_modified_envelope(
            *self.volume_envelope_params.clone(),
            control.envelope,
            self.stream_params.sample_rate as f32,
        );

        let allow_release = self.loop_params.mode != LoopMode::OneShot;

        let volume_envelope = SIMDVoiceEnvelope::new(
            *self.volume_envelope_params.clone(),
            modified_params,
            allow_release,
            self.stream_params.sample_rate as f32,
        );

        let amp = VoiceCombineSIMD::mult(volume_envelope, gen);
        amp
    }

    fn convert_to_voice<Gen>(&self, gen: Gen) -> Box<dyn Voice>
    where
        Gen: 'static + SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
    {
        let flattened = SIMDStereoVoice::new(gen);
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
        Gen: 'static + SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
    {
        let gen = self.apply_velocity(gen);
        let gen = self.apply_volume_oncc(gen, cc_state);
        let gen = self.apply_pan(gen);
        let gen = self.apply_envelope(gen, control);

        self.apply_cutoff_effect(gen)
    }

    /// MOVE FORK: live volume modulation via SFZ `volume_oncc<N>=<dB>`.
    /// The SIMDVoiceOnccAmp stage is always inserted to keep the
    /// generator chain monomorphic; when bindings are empty its
    /// `static_amp` short-circuit returns a permanent `set1(1.0)` and
    /// the cost matches the velocity `SIMDConstant` stage (one SIMD
    /// multiply per sample-width). On regions with non-empty bindings
    /// the stage polls the channel's CC atomic array on a counter
    /// (RECOMPUTE_INTERVAL — see voice/oncc_amp.rs) and produces the
    /// per-voice amp multiplier `db_to_amp(Σ delta·cc/127)`.
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
        let oncc = SIMDVoiceOnccAmp::<S>::new(cc_state.clone(), self.volume_oncc.clone());
        VoiceCombineSIMD::mult(oncc, gen)
    }

    fn apply_cutoff_effect(
        &self,
        gen: impl 'static + SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
    ) -> Box<dyn Voice> {
        if let Some(filter) = &self.filter {
            let gen = SIMDStereoVoiceCutoff::new(gen, filter);
            self.convert_to_voice(gen)
        } else {
            self.convert_to_voice(gen)
        }
    }
}

impl<S: 'static + Sync + Send + Simd> VoiceSpawner for StereoSampledVoiceSpawner<S> {
    fn spawn_voice(&self, control: &VoiceControlData, cc_state: &CcState) -> Box<dyn Voice> {
        self.begin_voice(control, cc_state)
    }

    fn exclusive_class(&self) -> Option<u8> {
        self.exclusive_class
    }
}
