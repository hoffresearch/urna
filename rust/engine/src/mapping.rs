//! `Mapping`: a read-only memory map of a whole file, the way a `.urna` is
//! read. `mmap`/`munmap` through `libc` on unix; `CreateFileMappingW`,
//! `MapViewOfFile` and `UnmapViewOfFile` from kernel32, declared here, on
//! windows. The mapping derefs to `[u8]` and is unmapped on drop; a
//! zero-length file maps to an empty slice without a system call (both
//! systems refuse to map zero bytes).
//!
//! Every `unsafe` of the mapping is in this file. Each block's `// SAFETY:`
//! names its invariant: the pointer and length are the ones the system
//! returned for a live view, and nothing writes through them.

use std::fs::File;
use std::io;
use std::ptr::NonNull;

/// A read-only view of a whole file.
pub struct Mapping {
    ptr: NonNull<u8>,
    len: usize,
}

impl std::fmt::Debug for Mapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // the length only, never the mapped bytes.
        f.debug_struct("Mapping")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

// SAFETY: the view is read-only for its whole life and `Mapping` hands out
// only shared `&[u8]` borrows of it, so moving it to another thread or
// reading it from several is the same as for a `Box<[u8]>`.
unsafe impl Send for Mapping {}
// SAFETY: as above; there is no interior mutability to race on.
unsafe impl Sync for Mapping {}

impl Mapping {
    /// Map all of `file` read-only.
    ///
    /// # Safety
    ///
    /// The file must not be truncated or written while the mapping lives:
    /// a shrink makes a read of the lost pages fault (SIGBUS) and a write
    /// changes bytes a caller already validated. Every mmap-based reader
    /// carries this contract; urna never rewrites a `.urna` in place.
    pub unsafe fn map(file: &File) -> io::Result<Self> {
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "file too large to map"))?;
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
            });
        }
        let ptr = sys::map(file, len)?;
        Ok(Self { ptr, len })
    }
}

impl std::ops::Deref for Mapping {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` is the start of a live read-only view of `len`
        // bytes (or dangling with `len == 0`, which a slice allows), and it
        // stays mapped until `drop`, which needs `&mut self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        if self.len > 0 {
            sys::unmap(self.ptr, self.len);
        }
    }
}

#[cfg(unix)]
mod sys {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::unix::io::AsRawFd;
    use std::ptr::NonNull;

    pub(super) fn map(file: &File, len: usize) -> io::Result<NonNull<u8>> {
        // SAFETY: a fresh read-only shared mapping of `len > 0` bytes of an
        // open descriptor at offset 0; the kernel picks the address and
        // reports failure as MAP_FAILED, checked below.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        NonNull::new(ptr.cast::<u8>()).ok_or_else(|| io::Error::other("mmap returned null"))
    }

    pub(super) fn unmap(ptr: NonNull<u8>, len: usize) {
        // SAFETY: `ptr` and `len` are exactly what `map` got from mmap, and
        // the owner is being dropped, so no borrow of the view outlives it.
        unsafe {
            libc::munmap(ptr.as_ptr().cast::<c_void>(), len);
        }
    }
}

#[cfg(windows)]
mod sys {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::windows::io::AsRawHandle;
    use std::ptr::NonNull;

    type Handle = *mut c_void;
    const PAGE_READONLY: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileMappingW(
            file: Handle,
            attributes: *mut c_void,
            protect: u32,
            size_high: u32,
            size_low: u32,
            name: *const u16,
        ) -> Handle;
        fn MapViewOfFile(
            mapping: Handle,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(base: *const c_void) -> i32;
        fn CloseHandle(handle: Handle) -> i32;
    }

    pub(super) fn map(file: &File, len: usize) -> io::Result<NonNull<u8>> {
        // SAFETY: a read-only, unnamed mapping object over the whole of an
        // open file handle (size 0 0 means the file's size); null is the
        // failure value, checked below.
        let mapping = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null_mut(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        if mapping.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: a read view of the first `len` bytes (the file's size) of
        // the live mapping object; null is the failure value.
        let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len) };
        let err = io::Error::last_os_error();
        // SAFETY: the mapping object's handle came from CreateFileMappingW
        // above and is closed once; an open view keeps the object alive.
        unsafe {
            CloseHandle(mapping);
        }
        NonNull::new(view.cast::<u8>()).ok_or(err)
    }

    pub(super) fn unmap(ptr: NonNull<u8>, _len: usize) {
        // SAFETY: `ptr` is the base address MapViewOfFile returned, and the
        // owner is being dropped, so no borrow of the view outlives it.
        unsafe {
            UnmapViewOfFile(ptr.as_ptr().cast::<c_void>());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Mapping;
    use std::io::Write;

    fn scratch(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("urna-mapping-{}-{name}", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(bytes)
            .unwrap();
        path
    }

    #[test]
    fn maps_a_file_and_an_empty_one() {
        let path = scratch("full", b"urna mapping bytes");
        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: the test owns the file and does not write it while mapped.
        let map = unsafe { Mapping::map(&file) }.unwrap();
        assert_eq!(&map[..], b"urna mapping bytes");
        let again = std::thread::spawn(move || map.len()).join().unwrap();
        assert_eq!(again, 18);
        let _ = std::fs::remove_file(&path);

        let path = scratch("empty", b"");
        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: as above.
        let map = unsafe { Mapping::map(&file) }.unwrap();
        assert!(map.is_empty());
        drop(map);
        let _ = std::fs::remove_file(&path);
    }
}
