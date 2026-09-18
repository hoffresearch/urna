//! bounded breadth-first traversal over `CsrIndex`. pure integer work: it
//! GENERATES candidate chunk ordinals and never scores. the caller hands the
//! deduped candidate set to the mandatory exact-cosine rerank
//! (`score_subset`), so a graph edge can never leak into a returned score.
//!
//! the visited set is a GENERATIONAL buffer (a `Vec<u32>` of stamps), not a
//! per-query `HashSet`: each call bumps a generation counter and marks a node
//! visited by writing the current generation into its slot. that makes
//! "already seen?" an O(1) integer compare with zero per-query allocation
//! beyond the frontier vectors, and the buffer is reused across hops.

use super::csr::CsrIndex;

/// reusable traversal scratch. allocate once per `CsrIndex`, reuse across
/// queries; `bounded_bfs` bumps the generation so stale stamps read as
/// unvisited without clearing the buffer.
pub struct Traversal {
    /// per-node last-seen generation stamp; `stamp[node] == cur_gen` means
    /// visited this query. 0 is the "never visited" sentinel, so the first
    /// real generation is 1.
    stamp: Vec<u32>,
    cur_gen: u32,
}

impl Traversal {
    pub fn new(n_nodes: usize) -> Self {
        Self {
            stamp: vec![0u32; n_nodes],
            cur_gen: 0,
        }
    }

    /// bounded bfs from `seeds`, expanding up to `hops` levels of the csr and
    /// capping the total returned candidate count at `max_frontier`. returns
    /// a deduped `Vec<usize>` of chunk ordinals (seeds first, in arrival
    /// order). pure integer, no scoring.
    pub fn bounded_bfs(
        &mut self,
        csr: &CsrIndex,
        seeds: &[usize],
        hops: usize,
        max_frontier: usize,
    ) -> Vec<usize> {
        // bump the generation; on the (astronomically unlikely) wraparound,
        // reset every stamp so a stale value cannot read as the new gen.
        self.cur_gen = self.cur_gen.wrapping_add(1);
        if self.cur_gen == 0 {
            for s in self.stamp.iter_mut() {
                *s = 0;
            }
            self.cur_gen = 1;
        }
        let cur_gen = self.cur_gen;
        let n = csr.n_nodes();

        let mut out: Vec<usize> = Vec::new();
        let mut frontier: Vec<usize> = Vec::new();

        for &s in seeds {
            if out.len() >= max_frontier {
                break;
            }
            if s >= n || s >= self.stamp.len() || self.stamp[s] == cur_gen {
                continue;
            }
            self.stamp[s] = cur_gen;
            out.push(s);
            frontier.push(s);
        }

        // expand `hops` levels; each level walks the current frontier's
        // neighbors, marking unseen ones, until the cap is hit.
        for _ in 0..hops {
            if out.len() >= max_frontier || frontier.is_empty() {
                break;
            }
            let mut next: Vec<usize> = Vec::new();
            'level: for &node in &frontier {
                for &nbr in csr.neighbors(node) {
                    if out.len() >= max_frontier {
                        break 'level;
                    }
                    let nbr = nbr as usize;
                    if nbr >= n || nbr >= self.stamp.len() || self.stamp[nbr] == cur_gen {
                        continue;
                    }
                    self.stamp[nbr] = cur_gen;
                    out.push(nbr);
                    next.push(nbr);
                }
            }
            frontier = next;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use urna_format::sections::graph::{EDGE_TYPE_NEXT_CHUNK, Edge, encode_graph_adjacency};

    /// a simple line graph 0->1->2->...->n-1 (NEXT_CHUNK forward) plus the
    /// reverse so bfs can walk both directions.
    fn line_csr(n: usize) -> CsrIndex {
        let mut edges = Vec::new();
        for i in 0..n.saturating_sub(1) {
            edges.push(Edge {
                src: i as u32,
                dst: (i + 1) as u32,
                edge_type: EDGE_TYPE_NEXT_CHUNK,
            });
            edges.push(Edge {
                src: (i + 1) as u32,
                dst: i as u32,
                edge_type: EDGE_TYPE_NEXT_CHUNK,
            });
        }
        let payload = encode_graph_adjacency(n, &edges).unwrap();
        CsrIndex::from_bytes(&payload, n).unwrap()
    }

    #[test]
    fn one_hop_collects_immediate_neighbors() {
        let csr = line_csr(10);
        let mut t = Traversal::new(10);
        // seed 5, 1 hop -> {5, 4, 6}.
        let mut got = t.bounded_bfs(&csr, &[5], 1, 100);
        got.sort_unstable();
        assert_eq!(got, vec![4, 5, 6]);
    }

    #[test]
    fn hops_bound_the_radius() {
        let csr = line_csr(20);
        let mut t = Traversal::new(20);
        // seed 10, 2 hops -> {10, 9, 11, 8, 12}.
        let mut got = t.bounded_bfs(&csr, &[10], 2, 100);
        got.sort_unstable();
        assert_eq!(got, vec![8, 9, 10, 11, 12]);
    }

    #[test]
    fn max_frontier_caps_the_candidate_count() {
        let csr = line_csr(50);
        let mut t = Traversal::new(50);
        let got = t.bounded_bfs(&csr, &[25], 10, 5);
        assert_eq!(got.len(), 5, "max_frontier must cap the result");
    }

    #[test]
    fn dedup_and_seeds_come_first() {
        let csr = line_csr(10);
        let mut t = Traversal::new(10);
        // overlapping seeds + neighbors must dedup; seeds preserved up front.
        let got = t.bounded_bfs(&csr, &[3, 3, 4], 1, 100);
        let mut sorted = got.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), got.len(), "result must be deduped");
        assert_eq!(&got[..2], &[3, 4], "seeds come first, in order");
    }

    #[test]
    fn generational_buffer_reuse_is_correct() {
        let csr = line_csr(10);
        let mut t = Traversal::new(10);
        // two queries reuse the same buffer; the second must not see stale
        // stamps from the first.
        let q1 = t.bounded_bfs(&csr, &[1], 1, 100);
        let q2 = t.bounded_bfs(&csr, &[8], 1, 100);
        let mut s1 = q1.clone();
        s1.sort_unstable();
        let mut s2 = q2.clone();
        s2.sort_unstable();
        assert_eq!(s1, vec![0, 1, 2]);
        assert_eq!(s2, vec![7, 8, 9]);
    }

    #[test]
    fn zero_hops_returns_only_seeds() {
        let csr = line_csr(10);
        let mut t = Traversal::new(10);
        let got = t.bounded_bfs(&csr, &[2, 5], 0, 100);
        assert_eq!(got, vec![2, 5]);
    }

    #[test]
    fn out_of_range_seed_is_ignored() {
        let csr = line_csr(5);
        let mut t = Traversal::new(5);
        let got = t.bounded_bfs(&csr, &[99, 2], 1, 100);
        let mut sorted = got.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3]);
    }
}
