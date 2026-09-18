//! `search_contract` section (`SECTION_SEARCH_CONTRACT = 0x06`). A
//! versioned JSON document mirroring the manifest's search-related
//! fields (metric, score_type, normalize, index_type, rerank_policy).
//! The reader cross-checks it against the manifest at open time; any
//! disagreement is rejected.

use super::codec::Cursor;
use crate::error::UrnaError;
use crate::layout::{
    SECTION_PAYLOAD_PREFIX_SIZE, SECTION_PAYLOAD_VERSION, SECTION_SEARCH_CONTRACT,
};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SearchContract {
    pub metric: String,
    pub score_type: String,
    pub normalize: String,
    pub index_type: String,
    pub rerank_policy: String,
}

pub fn encode_search_contract(contract: &SearchContract) -> crate::Result<Vec<u8>> {
    let json = serde_json::to_vec(contract)?;
    let json_len = u32::try_from(json.len())
        .map_err(|_| UrnaError::InvalidInput("search_contract JSON too large".into()))?;
    let mut buf = Vec::with_capacity(SECTION_PAYLOAD_PREFIX_SIZE + json.len());
    buf.extend_from_slice(&SECTION_PAYLOAD_VERSION.to_le_bytes());
    buf.extend_from_slice(&(json_len as u64).to_le_bytes());
    buf.extend_from_slice(&json);
    Ok(buf)
}

pub fn decode_search_contract(data: &[u8]) -> crate::Result<SearchContract> {
    let mut c = Cursor::new(data, SECTION_SEARCH_CONTRACT);
    if c.data.len() < SECTION_PAYLOAD_PREFIX_SIZE {
        return Err(c.malformed("payload shorter than prefix"));
    }
    let version = c.read_u32()?;
    if version != SECTION_PAYLOAD_VERSION {
        return Err(UrnaError::UnsupportedSectionVersion {
            section_id: SECTION_SEARCH_CONTRACT,
            version,
        });
    }
    let json_len = c.read_u64()? as usize;
    let json = c.read_bytes(json_len)?;
    let contract: SearchContract = serde_json::from_slice(json).map_err(UrnaError::Json)?;
    c.finish()?;
    Ok(contract)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let c = SearchContract {
            metric: "ip".into(),
            score_type: "cosine".into(),
            normalize: "l2".into(),
            index_type: "exact".into(),
            rerank_policy: "none".into(),
        };
        let bytes = encode_search_contract(&c).unwrap();
        let back = decode_search_contract(&bytes).unwrap();
        assert_eq!(c, back);
    }
}
