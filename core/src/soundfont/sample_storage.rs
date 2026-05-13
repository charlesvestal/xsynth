/* MOVE FORK: per-channel sample storage. Voices hit `.get(pos) -> i16` and
 * `.len()`; the storage backing those reads is either a normal heap Arc<[i16]>
 * (small one-off samples, SF2 path, fallback) or a mapped region inside a
 * `.x44c` cache file. The mapped variant is what lets two heavy DS libraries
 * (e.g. WörliTzer + Fender Bass) coexist on Move: the OS pages out idle
 * samples so resident memory tracks ACTIVE sample data, not total decoded
 * data. */

use std::sync::Arc;

/// Holds a memory-mapped region keep-alive token for one or more sample
/// channels that point into it. The OS unmaps the region when the Arc to
/// this holder drops to 0.
pub struct MmapHolder {
    pub mmap: memmap2::Mmap,
}

impl std::fmt::Debug for MmapHolder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MmapHolder({} bytes)", self.mmap.len())
    }
}

/// One channel of a decoded sample. Either heap-backed Arc<[i16]> or a
/// slice into an mmap-backed cache file.
#[derive(Clone)]
pub enum SampleStorage {
    Heap(Arc<[i16]>),
    Mapped {
        holder: Arc<MmapHolder>,
        byte_offset: usize,
        frames: usize,
    },
}

impl SampleStorage {
    #[inline(always)]
    pub fn get(&self, pos: usize) -> i16 {
        match self {
            SampleStorage::Heap(a) => match a.get(pos) {
                Some(&v) => v,
                None => 0,
            },
            SampleStorage::Mapped { holder, byte_offset, frames } => {
                if pos >= *frames { return 0; }
                let off = *byte_offset + pos * 2;
                let bytes = &holder.mmap[off..off + 2];
                i16::from_le_bytes([bytes[0], bytes[1]])
            }
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        match self {
            SampleStorage::Heap(a) => a.len(),
            SampleStorage::Mapped { frames, .. } => *frames,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for SampleStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleStorage::Heap(a) => write!(f, "Heap({})", a.len()),
            SampleStorage::Mapped { frames, .. } => write!(f, "Mapped({})", frames),
        }
    }
}
