//! OS-level mmap hints, carved out of `mmap_file.rs` so that file stays
//! under the 300-line crate guard.

use crate::mmap_file::MmapUrnaFile;

impl MmapUrnaFile {
    /// Hint to the OS that the mmap pages won't be needed soon. The
    /// next read will fault them back in from disk.
    ///
    /// **Caveat:** this is `posix_madvise(MADV_DONTNEED)` — an
    /// approximation of cold cache, NOT a guarantee. The OS may
    /// ignore the hint, keep pages around for prefetch, or return
    /// them from the kernel's page cache anyway. Use it for
    /// "madvise-cold" benchmarks (a useful upper bound on the warm
    /// case) but don't claim it's equivalent to a fresh boot. Real
    /// cold-cache benchmarks need `purge` (macOS) or `echo 3 >
    /// /proc/sys/vm/drop_caches` (Linux), both of which require
    /// privileged operations.
    #[cfg(unix)]
    pub fn madvise_cold(&self) {
        use std::ffi::c_void;
        // SAFETY: passing a valid mmap pointer + length. POSIX_MADV_DONTNEED
        // does not invalidate or move the mapping; we still hold the Mmap
        // and can read from it as before.
        unsafe {
            libc::posix_madvise(
                self._mmap.as_ptr() as *mut c_void,
                self._mmap.len(),
                libc::POSIX_MADV_DONTNEED,
            );
        }
    }

    /// No-op on non-Unix platforms.
    #[cfg(not(unix))]
    pub fn madvise_cold(&self) {}
}
