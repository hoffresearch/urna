//! A sorted list of strings with front coding: the first of every block of
//! 16 is stored whole, the rest as the length of the prefix shared with the
//! entry before plus the bytes after it. A payload rides with each entry.
//! Looking a prefix up is a binary search over the block heads and a walk
//! from there; walking decodes one entry after the other into one buffer.

use std::ops::Range;

use super::Unsorted;

const BLOCK: usize = 16;

#[derive(Default)]
pub struct Coded<T: Copy> {
    data: Vec<u8>,
    /// where each entry starts in `data`; one past the end last.
    offsets: Vec<u32>,
    payloads: Vec<T>,
}

impl<T: Copy> std::fmt::Debug for Coded<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Coded")
            .field("entries", &self.payloads.len())
            .field("bytes", &self.data.len())
            .finish()
    }
}

impl<T: Copy> Coded<T> {
    pub fn from_sorted<'a>(
        entries: impl IntoIterator<Item = (&'a str, T)>,
    ) -> Result<Self, Unsorted> {
        let mut data = Vec::new();
        let mut offsets = vec![0u32];
        let mut payloads = Vec::new();
        let mut prev = String::new();
        for (i, (s, payload)) in entries.into_iter().enumerate() {
            if i > 0 && s < prev.as_str() {
                return Err(Unsorted { at: i });
            }
            if i % BLOCK == 0 {
                data.extend_from_slice(s.as_bytes());
            } else {
                let lcp = common_prefix(&prev, s).min(255);
                data.push(u8::try_from(lcp).unwrap_or(255));
                data.extend_from_slice(&s.as_bytes()[lcp..]);
            }
            offsets.push(u32::try_from(data.len()).unwrap_or(u32::MAX));
            payloads.push(payload);
            prev.clear();
            prev.push_str(s);
        }
        Ok(Self {
            data,
            offsets,
            payloads,
        })
    }

    pub fn len(&self) -> usize {
        self.payloads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.payloads.is_empty()
    }

    pub fn payload(&self, i: usize) -> Option<T> {
        self.payloads.get(i).copied()
    }

    /// Entry `i`, decoded from its block.
    pub fn get(&self, i: usize) -> Option<String> {
        let mut out = None;
        self.walk(i, |j, s, _| {
            if j == i {
                out = Some(s.to_string());
            }
            j < i
        });
        out
    }

    /// The indices of the entries that start with `q`.
    pub fn prefix(&self, q: &str) -> Range<usize> {
        if q.is_empty() || self.is_empty() {
            return 0..0;
        }
        let blocks = self.len().div_ceil(BLOCK);
        // the last block whose head is not past q holds the first candidate.
        let block = (0..blocks)
            .collect::<Vec<_>>()
            .partition_point(|&b| self.head(b * BLOCK) < q)
            .saturating_sub(1);
        let mut start = self.len();
        let mut end = self.len();
        self.walk(block * BLOCK, |i, s, _| {
            if start == self.len() {
                if s >= q {
                    start = i;
                } else {
                    return true;
                }
            }
            if s.starts_with(q) {
                true
            } else {
                end = i;
                false
            }
        });
        if end <= start {
            return 0..0;
        }
        start..end
    }

    /// A block's first entry, stored whole.
    fn head(&self, i: usize) -> &str {
        let (a, b) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
        std::str::from_utf8(&self.data[a..b]).unwrap_or("")
    }

    /// Decodes the entries from the block of `from` on, calling `f` with
    /// each index, string and payload from `from` until it returns false.
    pub fn walk(&self, from: usize, mut f: impl FnMut(usize, &str, T) -> bool) {
        let mut buf: Vec<u8> = Vec::with_capacity(64);
        let mut i = from - from % BLOCK;
        while i < self.len() {
            let (a, b) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
            let entry = &self.data[a..b];
            if i % BLOCK == 0 {
                buf.clear();
                buf.extend_from_slice(entry);
            } else if let Some((&lcp, rest)) = entry.split_first() {
                buf.truncate(lcp as usize);
                buf.extend_from_slice(rest);
            }
            if i >= from
                && let Ok(s) = std::str::from_utf8(&buf)
                && !f(i, s, self.payloads[i])
            {
                return;
            }
            i += 1;
        }
    }
}

/// The length of the prefix `a` and `b` share, at a char boundary.
fn common_prefix(a: &str, b: &str) -> usize {
    let mut n = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
    while !a.is_char_boundary(n) || !b.is_char_boundary(n) {
        n -= 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_decode_whole_and_prefix_ranges_cross_blocks() {
        let mut v: Vec<String> = (0..50).map(|i| format!("ação {:02}", i % 23)).collect();
        v.push("zé".into());
        v.sort();
        let coded = Coded::from_sorted(v.iter().enumerate().map(|(i, s)| (s.as_str(), i)))
            .unwrap_or_default();
        assert_eq!(coded.len(), 51);
        for (i, s) in v.iter().enumerate() {
            assert_eq!(coded.get(i).as_deref(), Some(s.as_str()));
            assert_eq!(coded.payload(i), Some(i));
        }
        let r = coded.prefix("ação 1");
        assert_eq!(
            r.len(),
            v.iter().filter(|s| s.starts_with("ação 1")).count()
        );
        assert_eq!(coded.prefix("zé"), 50..51);
        assert_eq!(coded.prefix("z"), 50..51);
        assert_eq!(coded.prefix("b"), 0..0);
        assert_eq!(coded.prefix(""), 0..0);
        assert_eq!(common_prefix("açaí", "ação"), "aç".len());
    }
}
