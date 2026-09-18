//! JCS-canonical JSON serialization for the manifest. Field order
//! follows declaration order; `extra` keys are sorted (BTreeMap);
//! whitespace is stripped (CompactFormatter). Result is byte-identical
//! across machines for the same logical content.

use super::Manifest;
use crate::error::UrnaError;
use serde::Serialize;

impl Manifest {
    /// Serialize the manifest to canonical JSON: declaration order for the
    /// known fields, BTreeMap order for `extra`, no whitespace.
    pub fn to_canonical_json(&self) -> crate::Result<Vec<u8>> {
        let mut buf = Vec::new();
        let mut ser =
            serde_json::Serializer::with_formatter(&mut buf, serde_json::ser::CompactFormatter);
        self.serialize(&mut ser).map_err(UrnaError::Json)?;
        Ok(buf)
    }
}
