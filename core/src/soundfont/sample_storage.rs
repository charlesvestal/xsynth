/* MOVE FORK: per-channel sample storage. Voices hit `.get(pos) -> i16` and
 * `.len()`; the storage backing those reads is either a normal heap Arc<[i16]>
 * (small one-off samples, SF2 path, fallback) or a mapped region inside a
 * `.x44c` cache file. The mapped variant is what lets two heavy DS libraries
 * (e.g. WörliTzer + Fender Bass) coexist on Move: the OS pages out idle
 * samples so resident memory tracks ACTIVE sample data, not total decoded
 * data.
 *
 * Hot-path note: both backing modes ultimately point at i16 in stable
 * memory, so the public type is a single struct holding `(ptr, frames)`
 * directly. The enum match that used to live inside `get()` is gone — it's
 * one bounds check + one i16 load regardless of backing. The variant info
 * is kept only as a private `_keep` Arc to manage drop semantics. At ~70
 * voices × 8 reads/lane-pair × 128 frames the saved tag check is ~70 µs
 * per render block. */

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

/// Whatever Arc keeps the i16 buffer alive. Not touched on the hot path;
/// `ptr`/`frames` in `SampleStorage` are the live access pair.
enum Backing {
    Heap(Arc<[i16]>),
    Mapped(Arc<MmapHolder>),
}

impl Clone for Backing {
    fn clone(&self) -> Self {
        match self {
            Backing::Heap(a) => Backing::Heap(a.clone()),
            Backing::Mapped(h) => Backing::Mapped(h.clone()),
        }
    }
}

/// One channel of a decoded sample. `ptr..ptr+frames` is i16, read-only,
/// valid for the lifetime of the `_keep` Arc.
pub struct SampleStorage {
    _keep: Backing,
    ptr: *const i16,
    frames: usize,
}

// SAFETY: the *const i16 points into an Arc-managed allocation (heap slice
// or mmap region) that this SampleStorage keeps alive via `_keep`. The
// memory is read-only and stays mapped/valid for the lifetime of the Arc.
// Multiple voices share read access without synchronization.
unsafe impl Send for SampleStorage {}
unsafe impl Sync for SampleStorage {}

impl Clone for SampleStorage {
    fn clone(&self) -> Self {
        SampleStorage {
            _keep: self._keep.clone(),
            ptr: self.ptr,
            frames: self.frames,
        }
    }
}

impl SampleStorage {
    /// Heap-backed channel. The Arc keeps the slice alive; we cache the
    /// raw pointer so the hot path doesn't redo slice indexing.
    pub fn from_heap(buf: Arc<[i16]>) -> Self {
        let ptr = buf.as_ptr();
        let frames = buf.len();
        SampleStorage { _keep: Backing::Heap(buf), ptr, frames }
    }

    /// Construct a Mapped channel from an mmap region. The byte_offset
    /// must be 2-byte aligned (i16 alignment), which we guarantee at
    /// cache-write time (header is 28 bytes, channels start there).
    pub fn from_mmap(holder: Arc<MmapHolder>, byte_offset: usize, frames: usize) -> Self {
        let ptr = unsafe { holder.mmap.as_ptr().add(byte_offset) } as *const i16;
        SampleStorage { _keep: Backing::Mapped(holder), ptr, frames }
    }

    #[inline(always)]
    pub fn get(&self, pos: usize) -> i16 {
        if pos >= self.frames { return 0; }
        // SAFETY: pos < frames; ptr..ptr+frames is valid i16 for the
        // lifetime of `self._keep`.
        unsafe { *self.ptr.add(pos) }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.frames
    }

    pub fn is_empty(&self) -> bool {
        self.frames == 0
    }

    /// Raw access for SIMD bulk-read paths. `ptr..ptr+frames` is read-only
    /// i16, valid for the lifetime of `self`.
    #[inline(always)]
    pub fn raw(&self) -> (*const i16, usize) {
        (self.ptr, self.frames)
    }
}

impl std::fmt::Debug for SampleStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self._keep {
            Backing::Heap(_) => write!(f, "Heap({})", self.frames),
            Backing::Mapped(_) => write!(f, "Mapped({})", self.frames),
        }
    }
}

impl MmapHolder {
    /// MOVE FORK: hint to the kernel that this mmap will be read sparsely
    /// during voice playback. Skips the OS's normal read-ahead, which would
    /// waste page cache on samples whose notes won't play.
    pub fn advise_random(&self) {
        let _ = self.mmap.advise(memmap2::Advice::Random);
    }
}
