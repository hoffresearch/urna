//! visited-set abstraction for `layer_search`.
//!
//! the build loop calls `layer_search` once per layer per inserted node
//! (hundreds of thousands of calls on a 100k corpus) and used to allocate
//! a fresh `HashSet<u32>` for each. `VisitedList` is the classic hnswlib
//! trick: one `Vec<u32>` of epoch stamps sized `n`, allocated once per
//! build, cleared by bumping the epoch (o(1)), so a visit check is one
//! indexed load instead of a hash probe and the build has no per-call
//! allocation for it. the query path keeps a `HashSet` (a per-query
//! `vec![0; n]` would cost o(n) for an o(ef) walk).
//!
//! both implement `VisitSet`; the graph bytes do not depend on which one
//! is used (membership semantics are identical).

use std::collections::HashSet;

pub(super) trait VisitSet {
    /// mark `id` visited; `true` when it was not visited before.
    fn insert(&mut self, id: u32) -> bool;
}

impl VisitSet for HashSet<u32> {
    #[inline]
    fn insert(&mut self, id: u32) -> bool {
        HashSet::insert(self, id)
    }
}

pub(super) struct VisitedList {
    stamps: Vec<u32>,
    epoch: u32,
}

impl VisitedList {
    pub(super) fn new(n: usize) -> Self {
        Self {
            stamps: vec![0; n],
            epoch: 1,
        }
    }

    /// forget every visit. o(1) except once every 2^32 clears, when the
    /// stamps are reset so an old epoch can never read as current.
    #[inline]
    pub(super) fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamps.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
    }
}

impl VisitSet for VisitedList {
    #[inline]
    fn insert(&mut self, id: u32) -> bool {
        let slot = &mut self.stamps[id as usize];
        if *slot == self.epoch {
            false
        } else {
            *slot = self.epoch;
            true
        }
    }
}
