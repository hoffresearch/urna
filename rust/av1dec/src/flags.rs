//! The flag sets rav1d declared with the `bitflags` crate, now a
//! `macro_rules!` of ours with the same name and input (urna #643), so the
//! upstream declarations stay as written and only their `use` line moves.
//!
//! A declaration
//!
//! ```ignore
//! bitflags! {
//!     #[derive(Clone, Copy)]
//!     pub struct Needs: u8 {
//!         const LEFT = 1 << 0;
//!     }
//! }
//! ```
//!
//! becomes a tuple struct over the integer, one `const` per flag, the
//! `bitflags` 2 methods the decoder calls (all `const fn` but the `&mut`
//! ones) and the bit operators. The `impl Name: T { .. }` form adds the
//! constants and methods to a tuple struct declared beside it.

macro_rules! bitflags {
    (
        $(#[$outer:meta])*
        $vis:vis struct $name:ident: $t:ty {
            $(
                $(#[$inner:meta])*
                const $flag:ident = $value:expr;
            )*
        }
    ) => {
        $(#[$outer])*
        $vis struct $name($t);

        $crate::src::flags::bitflags! {
            impl $name: $t {
                $(
                    $(#[$inner])*
                    const $flag = $value;
                )*
            }
        }
    };
    (
        impl $name:ident: $t:ty {
            $(
                $(#[$inner:meta])*
                const $flag:ident = $value:expr;
            )*
        }
    ) => {
        #[allow(dead_code)]
        impl $name {
            $(
                $(#[$inner])*
                pub const $flag: Self = Self::from_bits_retain($value);
            )*

            pub const fn empty() -> Self {
                Self(0)
            }

            /// Every declared flag.
            #[allow(unused_mut)]
            pub const fn all() -> Self {
                let mut bits = 0;
                $(
                    bits |= Self::$flag.0;
                )*
                Self(bits)
            }

            pub const fn bits(&self) -> $t {
                self.0
            }

            pub const fn from_bits(bits: $t) -> Option<Self> {
                if (bits & !Self::all().0) == 0 {
                    Some(Self(bits))
                } else {
                    None
                }
            }

            /// The declared flags of `bits`, the others dropped.
            pub const fn from_bits_truncate(bits: $t) -> Self {
                Self(bits & Self::all().0)
            }

            /// `bits` as they are, undeclared ones included.
            pub const fn from_bits_retain(bits: $t) -> Self {
                Self(bits)
            }

            pub const fn is_empty(&self) -> bool {
                self.0 == 0
            }

            pub const fn is_all(&self) -> bool {
                (self.0 & Self::all().0) == Self::all().0
            }

            pub const fn contains(&self, other: Self) -> bool {
                (self.0 & other.0) == other.0
            }

            pub const fn intersects(&self, other: Self) -> bool {
                (self.0 & other.0) != 0
            }

            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }

            pub const fn intersection(self, other: Self) -> Self {
                Self(self.0 & other.0)
            }

            pub const fn difference(self, other: Self) -> Self {
                Self(self.0 & !other.0)
            }

            pub const fn symmetric_difference(self, other: Self) -> Self {
                Self(self.0 ^ other.0)
            }

            /// The declared flags not in `self`.
            pub const fn complement(self) -> Self {
                Self::from_bits_truncate(!self.0)
            }

            pub fn insert(&mut self, other: Self) {
                *self = self.union(other);
            }

            pub fn remove(&mut self, other: Self) {
                *self = self.difference(other);
            }

            pub fn toggle(&mut self, other: Self) {
                *self = self.symmetric_difference(other);
            }

            pub fn set(&mut self, other: Self, value: bool) {
                if value {
                    self.insert(other);
                } else {
                    self.remove(other);
                }
            }
        }

        impl ::std::ops::BitOr for $name {
            type Output = Self;

            fn bitor(self, other: Self) -> Self {
                self.union(other)
            }
        }

        impl ::std::ops::BitOrAssign for $name {
            fn bitor_assign(&mut self, other: Self) {
                self.insert(other);
            }
        }

        impl ::std::ops::BitAnd for $name {
            type Output = Self;

            fn bitand(self, other: Self) -> Self {
                self.intersection(other)
            }
        }

        impl ::std::ops::BitAndAssign for $name {
            fn bitand_assign(&mut self, other: Self) {
                *self = self.intersection(other);
            }
        }

        impl ::std::ops::BitXor for $name {
            type Output = Self;

            fn bitxor(self, other: Self) -> Self {
                self.symmetric_difference(other)
            }
        }

        impl ::std::ops::BitXorAssign for $name {
            fn bitxor_assign(&mut self, other: Self) {
                self.toggle(other);
            }
        }

        impl ::std::ops::Sub for $name {
            type Output = Self;

            fn sub(self, other: Self) -> Self {
                self.difference(other)
            }
        }

        impl ::std::ops::SubAssign for $name {
            fn sub_assign(&mut self, other: Self) {
                self.remove(other);
            }
        }

        impl ::std::ops::Not for $name {
            type Output = Self;

            fn not(self) -> Self {
                self.complement()
            }
        }
    };
}

pub(crate) use bitflags;
