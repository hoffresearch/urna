//! What rav1d derived with `strum`, written out (urna #643): the number of
//! variants of an enum and the variant of a discriminant. Each enum lists
//! its variants once, right after its definition, so a variant added there
//! and not here is a visible omission in the same diff.

/// The number of variants, as `strum::EnumCount` gave it.
pub trait EnumCount {
    const COUNT: usize;
}

/// `impl EnumCount` for a field-less enum, from the list of its variants.
macro_rules! enum_count {
    ($name:ident [$($variant:ident),* $(,)?]) => {
        impl $crate::src::enums::EnumCount for $name {
            const COUNT: usize = [$(stringify!($variant)),*].len();
        }
    };
}

pub(crate) use enum_count;

/// `const fn from_repr`, as `strum::FromRepr` derived it: the variant whose
/// discriminant is `discriminant` in the enum's `repr` type (`usize` when
/// the enum declares none), or `None`.
macro_rules! from_repr {
    ($name:ident: $repr:ty [$($variant:ident),* $(,)?]) => {
        impl $name {
            #[allow(dead_code)]
            pub const fn from_repr(discriminant: $repr) -> Option<Self> {
                $(
                    if discriminant == Self::$variant as $repr {
                        return Some(Self::$variant);
                    }
                )*
                None
            }
        }
    };
}

pub(crate) use from_repr;
