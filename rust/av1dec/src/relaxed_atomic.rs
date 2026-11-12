use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::sync::atomic::AtomicU16;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

/// An unsigned integer that has a std atomic of its width: the storage an
/// [`Atom`] packs into. (urna #643: this and [`Atom`]/[`Atomic`] replace the
/// `atomig` crate, with the same names so the call sites stay upstream's.)
pub trait AtomRepr: Copy {
    type Cell: Send + Sync;

    fn cell(value: Self) -> Self::Cell;
    fn load(cell: &Self::Cell, order: Ordering) -> Self;
    fn store(cell: &Self::Cell, value: Self, order: Ordering);
    fn compare_exchange(
        cell: &Self::Cell,
        current: Self,
        new: Self,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self, Self>;
}

macro_rules! atom_repr {
    ($($int:ty => $cell:ty),* $(,)?) => {$(
        impl AtomRepr for $int {
            type Cell = $cell;

            fn cell(value: Self) -> Self::Cell {
                <$cell>::new(value)
            }

            fn load(cell: &Self::Cell, order: Ordering) -> Self {
                cell.load(order)
            }

            fn store(cell: &Self::Cell, value: Self, order: Ordering) {
                cell.store(value, order)
            }

            fn compare_exchange(
                cell: &Self::Cell,
                current: Self,
                new: Self,
                success: Ordering,
                failure: Ordering,
            ) -> Result<Self, Self> {
                cell.compare_exchange(current, new, success, failure)
            }
        }
    )*};
}

atom_repr!(u8 => AtomicU8, u16 => AtomicU16, u32 => AtomicU32, u64 => AtomicU64);

/// A value that round-trips through an [`AtomRepr`], so it can live in an
/// [`Atomic`]. `unpack(pack(x)) == x` for every `x`.
pub trait Atom: Sized {
    type Repr: AtomRepr;

    fn pack(self) -> Self::Repr;
    fn unpack(src: Self::Repr) -> Self;
}

macro_rules! atom_cast {
    ($($t:ty => $repr:ty),* $(,)?) => {$(
        impl Atom for $t {
            type Repr = $repr;

            fn pack(self) -> $repr {
                self as $repr
            }

            fn unpack(src: $repr) -> Self {
                src as $t
            }
        }
    )*};
}

macro_rules! atom_self {
    ($($t:ty),* $(,)?) => {$(
        impl Atom for $t {
            type Repr = $t;

            fn pack(self) -> $t {
                self
            }

            fn unpack(src: $t) -> Self {
                src
            }
        }
    )*};
}

atom_self!(u8, u16, u32);
atom_cast!(i8 => u8, i16 => u16, i32 => u32);

impl Atom for bool {
    type Repr = u8;

    fn pack(self) -> u8 {
        self.into()
    }

    fn unpack(src: u8) -> Self {
        src != 0
    }
}

impl Atom for Option<NonZeroU32> {
    type Repr = u32;

    fn pack(self) -> u32 {
        self.map_or(0, NonZeroU32::get)
    }

    fn unpack(src: u32) -> Self {
        NonZeroU32::new(src)
    }
}

impl Atom for [i8; 4] {
    type Repr = u32;

    fn pack(self) -> u32 {
        u32::from_ne_bytes(self.map(|e| e as u8))
    }

    fn unpack(src: u32) -> Self {
        src.to_ne_bytes().map(|e| e as i8)
    }
}

impl Atom for [i16; 4] {
    type Repr = u64;

    fn pack(self) -> u64 {
        let [a, b, c, d] = self.map(i16::to_ne_bytes);
        u64::from_ne_bytes([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
    }

    fn unpack(src: u64) -> Self {
        let b = src.to_ne_bytes();
        [
            i16::from_ne_bytes([b[0], b[1]]),
            i16::from_ne_bytes([b[2], b[3]]),
            i16::from_ne_bytes([b[4], b[5]]),
            i16::from_ne_bytes([b[6], b[7]]),
        ]
    }
}

/// A `T` stored in the std atomic of its [`Atom::Repr`].
pub struct Atomic<T: Atom> {
    cell: <T::Repr as AtomRepr>::Cell,
    // `fn() -> T`: the cell holds no `T`, so `Send`/`Sync` follow the cell.
    marker: PhantomData<fn() -> T>,
}

impl<T: Atom> Atomic<T> {
    pub fn new(value: T) -> Self {
        Self {
            cell: <T::Repr as AtomRepr>::cell(value.pack()),
            marker: PhantomData,
        }
    }

    pub fn load(&self, order: Ordering) -> T {
        T::unpack(<T::Repr as AtomRepr>::load(&self.cell, order))
    }

    pub fn store(&self, value: T, order: Ordering) {
        <T::Repr as AtomRepr>::store(&self.cell, value.pack(), order)
    }

    pub fn compare_exchange(
        &self,
        current: T,
        new: T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<T, T> {
        <T::Repr as AtomRepr>::compare_exchange(
            &self.cell,
            current.pack(),
            new.pack(),
            success,
            failure,
        )
        .map(T::unpack)
        .map_err(T::unpack)
    }
}

impl<T: Atom + Default> Default for Atomic<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

/// A [`Atomic`] type that can only be accessed
/// through [`Ordering::Relaxed`] loads and stores,
/// which are essentially normal loads and stores.
///
/// This prevents the usage of methods like [`AtomicU8::fetch_or`]
/// that compile down to contended `lock *` instructions
/// even with [`Ordering::Relaxed`].
///
/// [`AtomicU8::fetch_or`]: std::sync::atomic::AtomicU8::fetch_or
#[derive(Default)]
pub struct RelaxedAtomic<T: Atom> {
    inner: Atomic<T>,
}

impl<T: Atom> RelaxedAtomic<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Atomic::new(value),
        }
    }

    pub fn get(&self) -> T {
        self.inner.load(Ordering::Relaxed)
    }

    pub fn set(&self, value: T) {
        self.inner.store(value, Ordering::Relaxed);
    }

    pub fn update(&self, f: impl Fn(T) -> T) {
        self.set(f(self.get()));
    }

    pub fn inner(&self) -> &Atomic<T> {
        &self.inner
    }
}

impl<T: Atom + Copy> RelaxedAtomic<T> {
    pub fn get_update(&self, f: impl Fn(T) -> T) -> T {
        let old = self.get();
        self.set(f(old));
        old
    }
}

impl<T: Atom> Clone for RelaxedAtomic<T> {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

impl<T: Atom> From<T> for RelaxedAtomic<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}
