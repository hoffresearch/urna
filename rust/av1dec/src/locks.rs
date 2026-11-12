//! The locks rav1d took from `parking_lot`, over `std::sync` (urna #643).
//!
//! The surface is the part of `parking_lot` the decoder calls, so the
//! upstream call sites stay as they are: `lock()` and `read()`/`write()`
//! return the guard, the `try_*` forms return an `Option`, and
//! [`Condvar::wait`] takes the guard by `&mut`. `parking_lot` has no lock
//! poisoning; a poisoned std lock here hands back its guard, which is what
//! `parking_lot` would have done (a panic while decoding already reached the
//! caller through the thread that panicked).

use std::fmt;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync;
use std::sync::PoisonError;
use std::sync::TryLockError;

pub type RwLockReadGuard<'a, T> = sync::RwLockReadGuard<'a, T>;
pub type RwLockWriteGuard<'a, T> = sync::RwLockWriteGuard<'a, T>;

fn ok_or_none<G>(result: Result<G, TryLockError<G>>) -> Option<G> {
    match result {
        Ok(guard) => Some(guard),
        Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

pub struct Mutex<T: ?Sized>(sync::Mutex<T>);

impl<T> Mutex<T> {
    pub const fn new(value: T) -> Self {
        Self(sync::Mutex::new(value))
    }

    pub fn into_inner(self) -> T {
        self.0.into_inner().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T: ?Sized> Mutex<T> {
    pub fn lock(&self) -> MutexGuard<'_, T> {
        MutexGuard(Some(self.0.lock().unwrap_or_else(PoisonError::into_inner)))
    }

    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        ok_or_none(self.0.try_lock()).map(|guard| MutexGuard(Some(guard)))
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T: Default> Default for Mutex<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Mutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// The std guard, held in an `Option` so [`Condvar::wait`] can move it out
/// and back through a `&mut` without `unsafe`; it is `None` only inside
/// that call.
pub struct MutexGuard<'a, T: ?Sized>(Option<sync::MutexGuard<'a, T>>);

impl<T: ?Sized> Deref for MutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.0.as_deref().unwrap()
    }
}

impl<T: ?Sized> DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0.as_deref_mut().unwrap()
    }
}

#[derive(Default)]
pub struct Condvar(sync::Condvar);

impl Condvar {
    pub const fn new() -> Self {
        Self(sync::Condvar::new())
    }

    /// Blocks until notified, releasing the lock while it waits, as
    /// `parking_lot::Condvar::wait` does.
    pub fn wait<T>(&self, guard: &mut MutexGuard<'_, T>) {
        let inner = guard.0.take().unwrap();
        guard.0 = Some(self.0.wait(inner).unwrap_or_else(PoisonError::into_inner));
    }

    pub fn notify_one(&self) {
        self.0.notify_one();
    }

    pub fn notify_all(&self) {
        self.0.notify_all();
    }
}

impl fmt::Debug for Condvar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

pub struct RwLock<T: ?Sized>(sync::RwLock<T>);

impl<T> RwLock<T> {
    pub const fn new(value: T) -> Self {
        Self(sync::RwLock::new(value))
    }

    pub fn into_inner(self) -> T {
        self.0.into_inner().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T: ?Sized> RwLock<T> {
    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        self.0.write().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn try_read(&self) -> Option<RwLockReadGuard<'_, T>> {
        ok_or_none(self.0.try_read())
    }

    pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, T>> {
        ok_or_none(self.0.try_write())
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T: Default> Default for RwLock<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for RwLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// One element of a read-locked `Vec`, the one use rav1d made of
/// `parking_lot::RwLockReadGuard::map` (std's `map` is still unstable).
pub struct ReadElement<'a, T> {
    guard: RwLockReadGuard<'a, Vec<T>>,
    index: usize,
}

impl<'a, T> ReadElement<'a, T> {
    pub fn new(guard: RwLockReadGuard<'a, Vec<T>>, index: usize) -> Self {
        // panics here, as the mapped guard did, not at the first deref
        assert!(index < guard.len(), "task index out of bounds");
        Self { guard, index }
    }
}

impl<T> Deref for ReadElement<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.guard[self.index]
    }
}
