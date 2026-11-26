//! Title pipelines the host driver's compiler is known to crash on, left out
//! of the run instead of ending it.
//!
//! On RDNA 4 (RX 9070 XT) the driver's compiler dies inside
//! `CreateComputePipelineState` (`amdxc64.dll+0x861b9f`, a read of `0x8`) on
//! some translated programs (issues #8, #44, #60). The early list then starts
//! the same pipeline at the next launch, and every later run crashes at
//! startup. `victory warm` already found such pipelines in processes of its
//! own and named them in `warm-bad.txt`, but the runs did not read it.
//!
//! Two lists beside the pipeline library, each line a pipeline key as
//! `victory warm` names it, then more words:
//! - `warm-bad.txt`: written by `victory warm` when the process making that
//!   pipeline ended or hung. Always left out.
//! - `crashed-pipelines.txt`: written by a run whose process ended with an
//!   exception on a thread while it was making that pipeline (the key, the
//!   driver's version, then why). Left out only while that driver is the
//!   one installed: a driver update tries it again.
//!
//! A pipeline left out is one a draw or dispatch goes without (the frame
//! line counts it as refused), which costs a pass instead of the run.
//! Delete a line (or the file) to try it again; `VICTORY_PIPELINE_QUARANTINE=0`
//! reads neither file and writes nothing.

use std::cell::Cell;
use std::collections::HashSet;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

/// Written by `victory warm`.
const WARM_BAD: &str = "warm-bad.txt";
/// Written by the runs.
const CRASHED: &str = "crashed-pipelines.txt";
/// A key as `Pending` makes it: 32 hex digits.
const KEY_LEN: usize = 32;

/// `VICTORY_PIPELINE_QUARANTINE=0`: off.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_QUARANTINE").is_none_or(|v| v != "0"))
}

/// The driver's version, read once while making pipelines (not in a crash:
/// the driver may hold its own locks there).
static DRIVER: OnceLock<i64> = OnceLock::new();

fn driver() -> i64 {
    *DRIVER.get_or_init(|| crate::host::host().map_or(0, |h| h.driver()))
}

thread_local! {
    /// The key of the pipeline this thread is making, if it is making one.
    static MAKING: Cell<Option<[u8; KEY_LEN]>> = const { Cell::new(None) };
}

/// While alive, this thread is making the pipeline with that key.
pub(crate) struct Making(Option<[u8; KEY_LEN]>);

impl Drop for Making {
    fn drop(&mut self) {
        let _ = MAKING.try_with(|m| m.set(self.0));
    }
}

/// Marks this thread as making the pipeline with `key` until the guard goes.
pub(crate) fn making(key: &str) -> Making {
    if !on() {
        return Making(MAKING.with(Cell::get));
    }
    if super::library::folder().is_some() {
        let _ = driver();
    }
    let mut bytes = [b'0'; KEY_LEN];
    let n = key.len().min(KEY_LEN);
    bytes[..n].copy_from_slice(&key.as_bytes()[..n]);
    Making(MAKING.with(|m| m.replace(Some(bytes))))
}

/// For the loader's crash report, on the thread that faulted: if it was
/// making a pipeline, that pipeline goes into `crashed-pipelines.txt`.
pub fn crashed_here() {
    let Some(key) = MAKING.try_with(Cell::get).ok().flatten() else {
        return;
    };
    let Some(dir) = super::library::folder() else {
        return;
    };
    let line = format!(
        "{} {} the process ended with an exception while this pipeline was being made\n",
        String::from_utf8_lossy(&key),
        DRIVER.get().copied().unwrap_or(0)
    );
    let path = dir.join(CRASHED);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(line.as_bytes());
        eprintln!(
            "VIctory: pipeline {} was being made on this thread; it is left out of later runs on this driver ({})",
            String::from_utf8_lossy(&key),
            path.display()
        );
    }
}

/// The keys left out, read once.
fn left_out_keys() -> &'static HashSet<String> {
    static KEYS: OnceLock<HashSet<String>> = OnceLock::new();
    KEYS.get_or_init(|| {
        let Some(dir) = super::library::folder().filter(|_| on()) else {
            return HashSet::new();
        };
        let read = |file| std::fs::read_to_string(dir.join(file)).unwrap_or_default();
        let (bad, crashed) = (read(WARM_BAD), read(CRASHED));
        let keys = parse(&bad, &crashed, driver());
        if !keys.is_empty() {
            tracing::warn!(
                pipelines = keys.len(),
                folder = %dir.display(),
                "title pipelines the driver's compiler crashed on are left out (warm-bad.txt, crashed-pipelines.txt; delete a line to try one again, VICTORY_PIPELINE_QUARANTINE=0 to make them all)"
            );
        }
        keys
    })
}

/// The keys of `warm-bad.txt` (every line) and of `crashed-pipelines.txt`
/// (lines of the driver `driver`).
fn parse(bad: &str, crashed: &str, driver: i64) -> HashSet<String> {
    let key = |word: &str| {
        (word.len() == KEY_LEN && word.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| word.to_owned())
    };
    let mut keys: HashSet<String> = bad
        .lines()
        .filter_map(|l| key(l.split(' ').next()?))
        .collect();
    keys.extend(crashed.lines().filter_map(|l| {
        let mut words = l.split(' ');
        let k = key(words.next()?)?;
        (words.next()?.parse::<i64>().ok()? == driver).then_some(k)
    }));
    keys
}

/// Whether the pipeline with `key` is left out (said once per key).
pub(crate) fn left_out(key: &str) -> bool {
    if !on() || !left_out_keys().contains(key) {
        return false;
    }
    static SAID: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
    if said.get_or_insert_default().insert(key.to_owned()) {
        tracing::warn!(
            key,
            "a title pipeline the driver's compiler crashed on is left out"
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_bad_keys_are_always_left_out_and_crashes_only_on_their_driver() {
        let a = "0123456789abcdef0123456789abcdef";
        let b = "fedcba9876543210fedcba9876543210";
        let c = "00112233445566778899aabbccddeeff";
        let bad = format!("{a} exit code: 0xc0000005\nnot a key\n\n");
        let crashed = format!("{b} 42 the process ended\n{c} 41 the process ended\n{c}\n");
        let keys = parse(&bad, &crashed, 42);
        assert!(keys.contains(a));
        assert!(keys.contains(b));
        assert!(
            !keys.contains(c),
            "a crash on another driver is tried again"
        );
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn the_guard_restores_what_the_thread_was_making() {
        assert_eq!(MAKING.with(Cell::get), None);
        {
            let _outer = making("0123456789abcdef0123456789abcdef");
            {
                let _inner = making("fedcba9876543210fedcba9876543210");
            }
            if on() {
                assert_eq!(MAKING.with(Cell::get).map(|k| k[0]), Some(b'0'));
            }
        }
        assert_eq!(MAKING.with(Cell::get), None);
    }
}
