//! `x.to::<T>()` for `x.into()` and `x.try_to::<T>()` for `x.try_into()`
//! with the target spelled out; what the `to_method` crate provided, kept
//! here so the fork has no dependency on it.

pub(crate) trait To {
    #[inline(always)]
    fn to<T>(self) -> T
    where
        Self: Into<T>,
    {
        self.into()
    }

    #[inline(always)]
    fn try_to<T>(self) -> Result<T, Self::Error>
    where
        Self: TryInto<T>,
    {
        self.try_into()
    }
}

impl<X> To for X {}
