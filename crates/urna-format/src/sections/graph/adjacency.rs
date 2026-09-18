//! chunk-to-chunk csr adjacency codec for SECTION_GRAPH_ADJACENCY (0x0C).
//!
//! OPTIONAL and EXCLUDED from content_hash (not in CANONICAL_SECTIONS), so
//! adding a graph never invalidates a urna:// citation. the wire encoding is
//! `raw`: one self-describing payload that reuses the shared `intpack` codec
//! (encoding id 4) for its integer columns, exactly like the hnsw section
//! (0x07). nothing here touches the embedding hot path. all integers le.
//!
//! payload: [u32 version=1][u64 n_nodes][intpack offsets, len n_nodes+1]
//! [intpack delta-gapped neighbor-id blob][edge-type column: u8 kind then
//! either one iso scalar (falkordb iso, kind 0) or an intpack u8 run of
//! len total_edges (kind 1)]. canonical edge order for byte-identical output
//! is ascending (src, edge_type, dst); [`encode_graph_adjacency`] sorts, so
//! two builds of the same edge set produce byte-identical bytes. typed
//! errors on truncation/hostile claims, never panics.

use crate::bytes::{le_u32, le_u64};
use crate::encoding::pack_u64s;
use crate::error::UrnaError;
use crate::layout::SECTION_GRAPH_ADJACENCY;

/// internal payload version for the csr layout. bumped on an internal
/// layout change, NEVER `URNA_FORMAT_VERSION` (the same lane-c discipline
/// as `HNSW_PAYLOAD_VERSION`).
pub const GRAPH_ADJACENCY_PAYLOAD_VERSION: u32 = 1;

/// edge type ids. a u8 column (iso single scalar when uniform).
pub const EDGE_TYPE_NEXT_CHUNK: u8 = 0;
pub const EDGE_TYPE_SEMANTIC: u8 = 1;
pub const EDGE_TYPE_CITATION: u8 = 2;

/// edge-type column kinds (the leading byte of the edge-type column).
const EDGE_COL_ISO: u8 = 0;
const EDGE_COL_RUN: u8 = 1;

/// hostile-input cap on a single node's out-degree. a real chunk graph
/// (NEXT_CHUNK + top-m semantic + citations) never approaches this, so it
/// never rejects a valid file; it bounds an untrusted decoded degree before
/// any allocation from a claim alone.
pub const MAX_DEGREE: u64 = 1 << 20;

/// one directed edge: `src -> dst` of a given `edge_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    pub src: u32,
    pub dst: u32,
    pub edge_type: u8,
}

fn malformed(reason: impl Into<String>) -> UrnaError {
    UrnaError::MalformedSectionPayload {
        section_id: SECTION_GRAPH_ADJACENCY,
        reason: reason.into(),
    }
}

/// encode `edges` over `n_nodes` chunk ordinals into the csr payload, byte-
/// identical for the same edge set (canonical sort). every `src`/`dst` < n.
pub fn encode_graph_adjacency(n_nodes: usize, edges: &[Edge]) -> Result<Vec<u8>, UrnaError> {
    for e in edges {
        if e.src as usize >= n_nodes || e.dst as usize >= n_nodes {
            return Err(malformed(format!(
                "edge ({},{}) out of range for n_nodes {}",
                e.src, e.dst, n_nodes
            )));
        }
    }
    // canonical order: ascending (src, edge_type, dst). byte-identical.
    let mut sorted = edges.to_vec();
    sorted.sort_unstable_by(|a, b| {
        a.src
            .cmp(&b.src)
            .then(a.edge_type.cmp(&b.edge_type))
            .then(a.dst.cmp(&b.dst))
    });

    // csr offsets (len n_nodes+1) + delta-gapped dst column + edge-type col.
    let mut offsets: Vec<u64> = Vec::with_capacity(n_nodes + 1);
    let mut dst_gaps: Vec<u64> = Vec::with_capacity(sorted.len());
    let mut types: Vec<u8> = Vec::with_capacity(sorted.len());
    offsets.push(0);
    let mut cursor = 0usize;
    for node in 0..n_nodes {
        // delta-gap dst within each (src, edge_type) run. the canonical sort
        // makes dst ascending inside a run, so every gap is non-negative; the
        // first dst of a run is absolute. resetting at each edge_type boundary
        // keeps decode a plain prefix-sum (no backward jumps across types).
        let mut prev_dst: Option<u32> = None;
        let mut prev_type: Option<u8> = None;
        while cursor < sorted.len() && sorted[cursor].src as usize == node {
            let e = sorted[cursor];
            if prev_type != Some(e.edge_type) {
                prev_dst = None;
                prev_type = Some(e.edge_type);
            }
            let gap = match prev_dst {
                Some(p) => (e.dst - p) as u64,
                None => e.dst as u64,
            };
            dst_gaps.push(gap);
            types.push(e.edge_type);
            prev_dst = Some(e.dst);
            cursor += 1;
        }
        offsets.push(dst_gaps.len() as u64);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&GRAPH_ADJACENCY_PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&(n_nodes as u64).to_le_bytes());
    push_intpack(&mut out, &offsets);
    push_intpack(&mut out, &dst_gaps);
    push_edge_types(&mut out, &types);
    Ok(out)
}

/// decode the csr payload back to its canonical edge list (the exact order
/// [`encode_graph_adjacency`] emitted: ascending src, edge_type, dst).
/// typed errors on any truncation or hostile claim; never panics.
pub fn decode_graph_adjacency(bytes: &[u8]) -> Result<(usize, Vec<Edge>), UrnaError> {
    let parts = parse_csr_parts(bytes)?;
    let mut edges = Vec::with_capacity(parts.neighbors.len());
    for node in 0..parts.n_nodes {
        let start = parts.offsets[node] as usize;
        let end = parts.offsets[node + 1] as usize;
        for i in start..end {
            edges.push(Edge {
                src: node as u32,
                dst: parts.neighbors[i],
                edge_type: parts.edge_types[i],
            });
        }
    }
    Ok((parts.n_nodes, edges))
}

fn push_intpack(out: &mut Vec<u8>, values: &[u64]) {
    let blob = pack_u64s(values);
    out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
    out.extend_from_slice(&blob);
}

fn push_edge_types(out: &mut Vec<u8>, types: &[u8]) {
    // iso single scalar when every edge shares one type (falkordb iso). an
    // empty edge set is iso with edge_type 0 (decode reads zero edges).
    let iso = types.first().copied().unwrap_or(0);
    if types.iter().all(|&t| t == iso) {
        out.push(EDGE_COL_ISO);
        out.push(iso);
    } else {
        out.push(EDGE_COL_RUN);
        let col: Vec<u64> = types.iter().map(|&t| t as u64).collect();
        push_intpack(out, &col);
    }
}

fn read_edge_types(cur: &mut Cursor, total: usize) -> Result<Vec<u8>, UrnaError> {
    let kind = cur.u8()?;
    match kind {
        EDGE_COL_ISO => {
            let iso = cur.u8()?;
            Ok(vec![iso; total])
        }
        EDGE_COL_RUN => {
            let col = cur.intpack_column()?;
            if col.len() != total {
                return Err(malformed("graph_adjacency: edge-type run length mismatch"));
            }
            col.iter()
                .map(|&v| {
                    u8::try_from(v).map_err(|_| malformed("graph_adjacency: edge-type > 255"))
                })
                .collect()
        }
        other => Err(malformed(format!(
            "graph_adjacency: unknown edge-type kind {}",
            other
        ))),
    }
}

/// light cursor over the csr payload. every read is bounds-checked and
/// returns a typed error, never a panic on a hostile mmap.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], UrnaError> {
        if n > self.buf.len() - self.pos {
            return Err(malformed("graph_adjacency: unexpected EOF"));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, UrnaError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, UrnaError> {
        le_u32(self.take(4)?)
    }
    fn u64(&mut self) -> Result<u64, UrnaError> {
        le_u64(self.take(8)?)
    }
    /// read a length-prefixed intpack blob, decode it to u64s.
    fn intpack_column(&mut self) -> Result<Vec<u64>, UrnaError> {
        let len = self.u32()? as usize;
        let blob = self.take(len)?;
        crate::encoding::unpack_u64s(blob)
    }
}

/// the validated csr columns the runtime `CsrIndex` indexes in O(1):
/// `offsets` (len n_nodes+1) bounds each node's run in `neighbors` (absolute
/// decoded dst ids) and `edge_types` (1:1 with `neighbors`).
pub struct CsrParts {
    pub n_nodes: usize,
    pub offsets: Vec<u64>,
    pub neighbors: Vec<u32>,
    pub edge_types: Vec<u8>,
}

/// parse the csr payload into validated owned columns (the single parser):
/// bounds-checks the header, monotone offsets, degree cap, column lengths,
/// and dst range, then de-gaps the neighbor ids into absolute ids so
/// `neighbors[start..end]` is a contiguous slice per node. never panics.
pub fn parse_csr_parts(bytes: &[u8]) -> Result<CsrParts, UrnaError> {
    let mut cur = Cursor::new(bytes);
    let version = cur.u32()?;
    if version != GRAPH_ADJACENCY_PAYLOAD_VERSION {
        return Err(UrnaError::UnsupportedSectionVersion {
            section_id: SECTION_GRAPH_ADJACENCY,
            version,
        });
    }
    let n_nodes = cur.u64()? as usize;
    let offsets = cur.intpack_column()?;
    // compare against offsets.len() (bounded by the physical payload) WITHOUT
    // computing n_nodes+1, which would overflow on a hostile n_nodes claim.
    if offsets.is_empty() || offsets.len() - 1 != n_nodes {
        return Err(malformed("graph_adjacency: offsets length != n_nodes+1"));
    }
    // offsets must be monotone non-decreasing; the last is the edge count.
    let mut total = 0u64;
    for w in offsets.windows(2) {
        if w[1] < w[0] {
            return Err(malformed("graph_adjacency: offsets not monotone"));
        }
        let deg = w[1] - w[0];
        if deg > MAX_DEGREE {
            return Err(malformed("graph_adjacency: degree exceeds cap"));
        }
        total = total.saturating_add(deg);
    }
    if offsets.last().copied() != Some(total) {
        return Err(malformed("graph_adjacency: final offset != edge count"));
    }
    let dst_gaps = cur.intpack_column()?;
    if dst_gaps.len() as u64 != total {
        return Err(malformed("graph_adjacency: dst column length mismatch"));
    }
    let edge_types = read_edge_types(&mut cur, total as usize)?;

    // de-gap per (node, edge_type) run into absolute dst ids, mirroring the
    // encoder's reset at each edge_type boundary, range-checking each.
    let mut neighbors = Vec::with_capacity(total as usize);
    for node in 0..n_nodes {
        let start = offsets[node] as usize;
        let end = offsets[node + 1] as usize;
        let mut prev_dst: Option<u32> = None;
        let mut prev_type: Option<u8> = None;
        for i in start..end {
            let t = edge_types[i];
            if prev_type != Some(t) {
                prev_dst = None;
                prev_type = Some(t);
            }
            let dst = match prev_dst {
                Some(p) => (p as u64).wrapping_add(dst_gaps[i]) as u32,
                None => dst_gaps[i] as u32,
            };
            if dst as usize >= n_nodes {
                return Err(malformed("graph_adjacency: dst out of range"));
            }
            neighbors.push(dst);
            prev_dst = Some(dst);
        }
    }
    Ok(CsrParts {
        n_nodes,
        offsets,
        neighbors,
        edge_types,
    })
}
