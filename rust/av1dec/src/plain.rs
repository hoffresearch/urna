//! The three marker traits rav1d took from `zerocopy` 0.7, and the byte
//! conversions it calls on them, written out here (urna #643). The names
//! and signatures are zerocopy's, so the call sites stay upstream's.
//!
//! The derives are replaced by [`plain!`], which writes the `unsafe impl`s
//! and checks at compile time what the derive checked: every field
//! implements the trait, and the fields fill the struct (no padding, for
//! [`AsBytes`]). Each such struct is `#[repr(C)]` or `#[repr(transparent)]`
//! (or has one field, for [`FromZeroes`] alone), as the derive required.

use std::mem;
use std::slice;

/// # Safety
///
/// The all-zero byte pattern is a valid `Self`.
pub unsafe trait FromZeroes {
    fn new_zeroed() -> Self
    where
        Self: Sized,
    {
        // SAFETY: the all-zero pattern is a valid `Self`, the trait's contract.
        unsafe { mem::zeroed() }
    }
}

fn fits<T>(bytes: &[u8]) -> bool {
    bytes.len() == mem::size_of::<T>() && bytes.as_ptr().cast::<T>().is_aligned()
}

fn fits_slice<T>(bytes: &[u8]) -> Option<usize> {
    let size = mem::size_of::<T>();
    let aligned = bytes.as_ptr().cast::<T>().is_aligned();
    (size != 0 && aligned && bytes.len() % size == 0).then(|| bytes.len() / size)
}

/// # Safety
///
/// Every byte pattern of the size of a `Self` is a valid `Self`.
pub unsafe trait FromBytes: FromZeroes {
    /// `bytes` as a `Self`, if it has exactly the size and alignment.
    fn ref_from(bytes: &[u8]) -> Option<&Self>
    where
        Self: Sized,
    {
        if !fits::<Self>(bytes) {
            return None;
        }
        // SAFETY: `fits` checked the size and alignment, every pattern is a
        // valid `Self` (`FromBytes`), and the borrow keeps `bytes`' lifetime.
        Some(unsafe { &*bytes.as_ptr().cast::<Self>() })
    }

    fn mut_from(bytes: &mut [u8]) -> Option<&mut Self>
    where
        Self: Sized + AsBytes,
    {
        if !fits::<Self>(bytes) {
            return None;
        }
        // SAFETY: as in `ref_from`, and the `&mut` is unique; any value
        // written through it is plain bytes again (`AsBytes`).
        Some(unsafe { &mut *bytes.as_mut_ptr().cast::<Self>() })
    }

    /// The first `size_of::<Self>()` bytes as a `Self`.
    fn ref_from_prefix(bytes: &[u8]) -> Option<&Self>
    where
        Self: Sized,
    {
        Self::ref_from(bytes.get(..mem::size_of::<Self>())?)
    }

    fn mut_from_prefix(bytes: &mut [u8]) -> Option<&mut Self>
    where
        Self: Sized + AsBytes,
    {
        Self::mut_from(bytes.get_mut(..mem::size_of::<Self>())?)
    }

    /// `bytes` as a slice of `Self`, if the length is a multiple of the size
    /// and the start is aligned.
    fn slice_from(bytes: &[u8]) -> Option<&[Self]>
    where
        Self: Sized,
    {
        let len = fits_slice::<Self>(bytes)?;
        // SAFETY: `len` elements of `Self` cover exactly `bytes`, which is
        // aligned for `Self`; every pattern is valid (`FromBytes`).
        Some(unsafe { slice::from_raw_parts(bytes.as_ptr().cast::<Self>(), len) })
    }

    fn mut_slice_from(bytes: &mut [u8]) -> Option<&mut [Self]>
    where
        Self: Sized + AsBytes,
    {
        let len = fits_slice::<Self>(bytes)?;
        // SAFETY: as in `slice_from`, and the `&mut` is unique.
        Some(unsafe { slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<Self>(), len) })
    }
}

/// # Safety
///
/// A `Self` has no padding and no uninitialized byte.
pub unsafe trait AsBytes {
    fn as_bytes(&self) -> &[u8] {
        let len = mem::size_of_val(self);
        // SAFETY: all `len` bytes of `self` are initialized (`AsBytes`),
        // `u8` has alignment 1, and the borrow keeps `self`'s lifetime.
        unsafe { slice::from_raw_parts((self as *const Self).cast::<u8>(), len) }
    }

    fn as_bytes_mut(&mut self) -> &mut [u8]
    where
        Self: FromBytes,
    {
        let len = mem::size_of_val(self);
        // SAFETY: as in `as_bytes`, the `&mut` is unique, and any bytes
        // written are a valid `Self` (`FromBytes`).
        unsafe { slice::from_raw_parts_mut((self as *mut Self).cast::<u8>(), len) }
    }
}

macro_rules! plain_primitive {
    ($($t:ty),*) => {$(
        // SAFETY: an integer or a float: no padding, every bit pattern valid.
        unsafe impl FromZeroes for $t {}
        // SAFETY: as above.
        unsafe impl FromBytes for $t {}
        // SAFETY: as above.
        unsafe impl AsBytes for $t {}
    )*};
}

// SAFETY: the integers and floats have no padding and no invalid pattern.
plain_primitive!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

// SAFETY: `false` is the zero byte.
unsafe impl FromZeroes for bool {}
// SAFETY: one initialized byte.
unsafe impl AsBytes for bool {}

// SAFETY: an array or a slice is its elements back to back, no padding
// between them; it holds the property its element holds.
unsafe impl<T: FromZeroes, const N: usize> FromZeroes for [T; N] {}
// SAFETY: as above.
unsafe impl<T: FromBytes, const N: usize> FromBytes for [T; N] {}
// SAFETY: as above.
unsafe impl<T: AsBytes, const N: usize> AsBytes for [T; N] {}
// SAFETY: as above.
unsafe impl<T: FromZeroes> FromZeroes for [T] {}
// SAFETY: as above.
unsafe impl<T: FromBytes> FromBytes for [T] {}
// SAFETY: as above.
unsafe impl<T: AsBytes> AsBytes for [T] {}

/// The sum of `sizes`, in a `const` context.
pub const fn sum(sizes: &[usize]) -> usize {
    let mut total = 0;
    let mut i = 0;
    while i < sizes.len() {
        total += sizes[i];
        i += 1;
    }
    total
}

/// `unsafe impl` of the listed traits for a struct, from the list of its
/// field types: each field must implement each trait, and the fields must
/// add up to the struct's size (no padding), both checked at compile time.
macro_rules! plain {
    ($name:ty: [$($trait:ident),+] = $fields:tt) => {
        const _: () = $crate::src::plain::plain!(@size $name $fields);
        $(
            // SAFETY: `@one` checks the fields and states the invariant.
            $crate::src::plain::plain!(@one $trait $name $fields);
        )+
    };
    (@size $name:ty [$($field:ty),* $(,)?]) => {
        assert!(
            ::std::mem::size_of::<$name>()
                == $crate::src::plain::sum(&[$(::std::mem::size_of::<$field>()),*])
        )
    };
    (@one $trait:ident $name:ty [$($field:ty),* $(,)?]) => {
        const _: () = {
            fn check<T: ?Sized + $crate::src::plain::$trait>() {}
            $( let _ = check::<$field>; )*
        };
        // SAFETY: the struct is `repr(C)` or `repr(transparent)` (or a single
        // field), each field implements the trait, and the size check above
        // proves there is no padding: the struct holds what its fields hold.
        unsafe impl $crate::src::plain::$trait for $name {}
    };
}

pub(crate) use plain;

// levels.rs denies `unsafe_code` (upstream's choice), so the impls for its
// types, which zerocopy's derives wrote there, live here.
mod levels {
    use super::FromZeroes;
    use crate::src::levels::Av1BlockInter1d;
    use crate::src::levels::Av1BlockInter2d;
    use crate::src::levels::BlockSize;
    use crate::src::levels::MaskedInterIntraPredMode;
    use crate::src::levels::Mv;

    const _: () = assert!(BlockSize::Bs128x128 as u8 == 0);
    // SAFETY: `BlockSize` is `repr(u8)` and `Bs128x128 = 0`, so the zero
    // byte is a variant.
    unsafe impl FromZeroes for BlockSize {}

    // SAFETY: `plain!` checks the fields and the size; the struct is `repr(C)` or `repr(transparent)`.
    super::plain!(Mv: [FromZeroes, FromBytes, AsBytes] = [i16, i16]);
    // SAFETY: `plain!` checks the fields and the size; the struct is `repr(C)` or `repr(transparent)`.
    super::plain!(MaskedInterIntraPredMode: [FromZeroes, FromBytes, AsBytes] = [u8]);
    // SAFETY: `plain!` checks the fields and the size; the struct is `repr(C)` or `repr(transparent)`.
    super::plain!(Av1BlockInter1d: [FromZeroes, FromBytes, AsBytes] = [[Mv; 2], u8, u8, MaskedInterIntraPredMode, u8]);
    // SAFETY: `plain!` checks the fields and the size; the struct is `repr(C)` or `repr(transparent)`.
    super::plain!(Av1BlockInter2d: [FromZeroes, FromBytes, AsBytes] = [Mv, [i16; 4]]);
}
