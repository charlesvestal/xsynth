use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::sync::Arc;

/// MOVE FORK: convert an f32 sample (nominally -1.0..=1.0) to i16. Clamps
/// out-of-range values to the i16 limits so a hot kick drum doesn't wrap
/// around to a negative value.
#[inline(always)]
fn f32_to_i16(s: f32) -> i16 {
    let v = (s * 32767.0).round();
    if v <= i16::MIN as f32 { i16::MIN }
    else if v >= i16::MAX as f32 { i16::MAX }
    else { v as i16 }
}

/// MOVE FORK: resample then store as i16 to halve the in-memory footprint
/// of the decoded sample pool. xsynth's f32 storage was costing ~950 MB for
/// WörliTzer alone; i16 brings that to ~475 MB so multi-track sets with
/// heavy DS libraries fit on Move's 1.85 GB device. Quality loss is bounded
/// to 16-bit dynamic range (still CD-quality); the input samples were
/// typically 16-bit on disk anyway.
pub fn resample_vecs(
    vecs: Vec<Vec<f32>>,
    sample_rate: f32,
    new_sample_rate: f32,
) -> Arc<[Arc<[i16]>]> {
    vecs.into_iter()
        .map(|samples| resample_vec(samples, sample_rate, new_sample_rate))
        .collect()
}

pub fn resample_vec(vec: Vec<f32>, sample_rate: f32, new_sample_rate: f32) -> Arc<[i16]> {
    let f32_out: Vec<f32> = if (sample_rate - new_sample_rate).abs() < 1e-3 {
        // No resample needed — pass-through.
        vec
    } else {
        let params = SincInterpolationParameters {
            sinc_len: 32,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };
        let len = vec.len();
        let mut resampler = SincFixedIn::<f32>::new(
            new_sample_rate as f64 / sample_rate as f64,
            2.0,
            params,
            len,
            1,
        )
        .unwrap();
        resampler.process(&[vec], None).unwrap().pop().unwrap_or_default()
    };
    f32_out.into_iter().map(f32_to_i16).collect()
}
