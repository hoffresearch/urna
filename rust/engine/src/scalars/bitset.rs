//! One bit per chunk: the set a filter keeps. 38 thousand chunks are 4.8
//! KB, so every predicate is a pass over a column and an `and` is a word
//! operation.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bitset {
    words: Vec<u64>,
    n: usize,
}

impl Bitset {
    pub fn empty(n: usize) -> Self {
        Self {
            words: vec![0; n.div_ceil(64)],
            n,
        }
    }

    pub fn full(n: usize) -> Self {
        let mut s = Self::empty(n);
        for i in 0..n {
            s.set(i);
        }
        s
    }

    pub fn from_fn(n: usize, keep: impl Fn(usize) -> bool) -> Self {
        let mut s = Self::empty(n);
        for i in 0..n {
            if keep(i) {
                s.set(i);
            }
        }
        s
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn set(&mut self, i: usize) {
        if i < self.n {
            self.words[i / 64] |= 1 << (i % 64);
        }
    }

    pub fn get(&self, i: usize) -> bool {
        i < self.n && self.words[i / 64] & (1 << (i % 64)) != 0
    }

    /// how many chunks are kept.
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// every chunk kept, in order.
    pub fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.n).filter(move |&i| self.get(i))
    }

    pub fn and(&mut self, other: &Bitset) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= *b;
        }
    }

    pub fn not(&mut self) {
        for w in &mut self.words {
            *w = !*w;
        }
        // clear the bits past n in the last word.
        let tail = self.n % 64;
        if tail != 0
            && let Some(last) = self.words.last_mut()
        {
            *last &= (1u64 << tail) - 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_count_and_not_respect_the_length() {
        let mut s = Bitset::empty(70);
        s.set(0);
        s.set(69);
        s.set(70);
        assert!(s.get(0) && s.get(69) && !s.get(70) && !s.get(1));
        assert_eq!(s.count(), 2);
        assert_eq!(s.ones().collect::<Vec<_>>(), vec![0, 69]);
        s.not();
        assert_eq!(s.count(), 68);
        let mut all = Bitset::full(70);
        all.and(&s);
        assert_eq!(all.count(), 68);
        assert_eq!(Bitset::from_fn(5, |i| i % 2 == 0).count(), 3);
    }
}
