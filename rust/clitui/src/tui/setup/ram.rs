//! The machine's total ram, for the model screen's fit mark: one probe per
//! system, read from what the system already publishes, no crate. `None`
//! when the system is none of these or the probe fails.

/// Total physical memory in bytes.
pub fn total() -> Option<u64> {
    probe().filter(|n| *n > 0)
}

/// macOS: `sysctl -n hw.memsize` prints the bytes.
#[cfg(target_os = "macos")]
fn probe() -> Option<u64> {
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// Linux: the `MemTotal` line of `/proc/meminfo`, in kibibytes.
#[cfg(target_os = "linux")]
fn probe() -> Option<u64> {
    meminfo_total(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

#[cfg(any(target_os = "linux", test))]
fn meminfo_total(meminfo: &str) -> Option<u64> {
    let rest = meminfo.lines().find_map(|l| l.strip_prefix("MemTotal:"))?;
    let mut words = rest.split_whitespace();
    let kib: u64 = words.next()?.parse().ok()?;
    match words.next() {
        Some("kB") | None => kib.checked_mul(1024),
        Some(_) => None,
    }
}

/// Windows: `GlobalMemoryStatusEx` from kernel32, declared here.
#[cfg(windows)]
fn probe() -> Option<u64> {
    /// MEMORYSTATUSEX: the system fills every field; only the total is read.
    #[repr(C)]
    #[allow(dead_code, reason = "the layout the system writes; one field is read")]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    // SAFETY: the signature is kernel32's own (BOOL GlobalMemoryStatusEx(
    // LPMEMORYSTATUSEX)), with the struct above laid out as the C one.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    // SAFETY: `status` is a live, writable MEMORYSTATUSEX with `length` set
    // to its size, which is all the call requires; it writes only into it.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    (ok != 0).then_some(status.total_phys)
}

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
fn probe() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meminfo_reads_the_total_in_bytes() {
        let text = "MemTotal:       16303060 kB\nMemFree:         1000 kB\n";
        assert_eq!(meminfo_total(text), Some(16_303_060 * 1024));
        assert_eq!(meminfo_total("MemFree: 1 kB\n"), None);
        assert_eq!(meminfo_total("MemTotal: lots kB\n"), None);
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux", windows))]
    fn this_machine_reports_some_ram() {
        let bytes = total().unwrap();
        assert!(bytes > 256 * 1024 * 1024, "{bytes}");
    }
}
