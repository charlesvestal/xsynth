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
///
/// MOVE FORK perf note: Mapped variant precomputes a typed i16 pointer
/// into the mmap region so per-sample get() is one bounds check + one
/// aligned i16 load — same hot-path cost as Arc<[i16]>. The earlier
/// byte-arithmetic version pushed render time to 12 µs/voice; the typed
/// pointer drops that back near 6 µs/voice. The pointer is kept alive
/// by the Arc<MmapHolder> sharing this SampleStorage.
pub enum SampleStorage {
    Heap(Arc<[i16]>),
    Mapped {
        holder: Arc<MmapHolder>,
        ptr: *const i16,
        frames: usize,
    },
}

// SAFETY: the *const i16 in `Mapped` points into the MmapHolder we also
// hold an Arc to; the mmap region is read-only and stays mapped for the
// lifetime of the Arc. Multiple voices share read access without
// synchronization.
unsafe impl Send for SampleStorage {}
unsafe impl Sync for SampleStorage {}

impl Clone for SampleStorage {
    fn clone(&self) -> Self {
        match self {
            SampleStorage::Heap(a) => SampleStorage::Heap(a.clone()),
            SampleStorage::Mapped { holder, ptr, frames } => SampleStorage::Mapped {
                holder: holder.clone(),
                ptr: *ptr,
                frames: *frames,
            },
        }
    }
}

impl SampleStorage {
    /// Construct a Mapped channel from an mmap region. The byte_offset
    /// must be 2-byte aligned (i16 alignment), which we guarantee at
    /// cache-write time (header is 28 bytes, channels start there).
    pub fn from_mmap(holder: Arc<MmapHolder>, byte_offset: usize, frames: usize) -> Self {
        let ptr = unsafe { holder.mmap.as_ptr().add(byte_offset) } as *const i16;
        SampleStorage::Mapped { holder, ptr, frames }
    }

    #[inline(always)]
    pub fn get(&self, pos: usize) -> i16 {
        match self {
            SampleStorage::Heap(a) => match a.get(pos) {
                Some(&v) => v,
                None => 0,
            },
            SampleStorage::Mapped { ptr, frames, .. } => {
                if pos >= *frames { return 0; }
                // SAFETY: pos < frames and the Arc<MmapHolder> kept by this
                // SampleStorage guarantees `ptr..ptr+frames` is mapped.
                unsafe { *ptr.add(pos) }
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

impl MmapHolder {
    /// MOVE FORK: hint to the kernel that this mmap will be read sequentially
    /// during voice playback. Skips read-ahead aggressiveness that's wasteful
    /// when most pages will only be touched if their note plays.
    pub fn advise_random(&self) {
        let _ = self.mmap.advise(memmap2::Advice::Random);
    }
}
