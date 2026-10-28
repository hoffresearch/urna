//! A safetensors file read here: 8-byte little-endian header length, a
//! json header naming each tensor's dtype, shape and byte range, then the
//! payload. Every claim is bounded by the file before it is used, and a
//! git-lfs pointer left in the table's place is named as such.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::{InferError, Result};

#[derive(Clone, Debug, Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// The file's bytes and its parsed header.
pub struct SafeTensors {
    path: String,
    bytes: Vec<u8>,
    data_start: usize,
    tensors: BTreeMap<String, TensorInfo>,
}

impl std::fmt::Debug for SafeTensors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafeTensors")
            .field("path", &self.path)
            .field("tensors", &self.tensors.len())
            .finish_non_exhaustive()
    }
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(bytes, &path.display().to_string())
    }

    pub fn from_bytes(bytes: Vec<u8>, path: &str) -> Result<Self> {
        let fail = |reason: &str| InferError::SafeTensors {
            path: path.to_string(),
            reason: reason.to_string(),
        };
        if bytes.starts_with(b"version https://git-lfs") {
            return Err(InferError::LfsPointer {
                path: path.to_string(),
            });
        }
        let head: [u8; 8] = bytes
            .get(..8)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| fail("truncated"))?;
        let n = u64::from_le_bytes(head) as usize;
        let end = 8usize
            .checked_add(n)
            .ok_or_else(|| fail("header past the file"))?;
        let header = bytes
            .get(8..end)
            .ok_or_else(|| fail("header past the file"))?;
        let mut raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(header)?;
        raw.remove("__metadata__");
        let mut tensors = BTreeMap::new();
        let data_len = bytes.len() - 8 - n;
        for (name, v) in raw {
            let info: TensorInfo = serde_json::from_value(v)?;
            let [a, b] = info.data_offsets;
            if a > b || b > data_len {
                return Err(fail(&format!("tensor {name}: data range past the file")));
            }
            let elems: usize = info
                .shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| fail(&format!("tensor {name}: shape overflows")))?;
            let width = match info.dtype.as_str() {
                "F32" => 4,
                "F16" | "BF16" => 2,
                _ => 1,
            };
            if info.dtype != "F32" && info.dtype != "F16" && info.dtype != "BF16" {
                continue;
            }
            if elems.checked_mul(width) != Some(b - a) {
                return Err(fail(&format!(
                    "tensor {name}: shape does not match its bytes"
                )));
            }
            tensors.insert(name, info);
        }
        Ok(Self {
            path: path.to_string(),
            bytes,
            data_start: 8 + n,
            tensors,
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    /// The tensor as f32, converting f16 and bf16 rows on the way.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| InferError::SafeTensors {
                path: self.path.clone(),
                reason: format!("no tensor {name}"),
            })?;
        let [a, b] = info.data_offsets;
        let data = &self.bytes[self.data_start + a..self.data_start + b];
        Ok(match info.dtype.as_str() {
            "F32" => data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            "F16" => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect(),
            "BF16" => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_le_bytes(*c).to_f32())
                .collect(),
            other => {
                return Err(InferError::SafeTensors {
                    path: self.path.clone(),
                    reason: format!("tensor {name}: dtype {other} is not a float"),
                });
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn reads_f32_and_f16_tensors_and_refuses_bad_claims() {
        let mut data = Vec::new();
        for x in [1.0f32, 2.0, 3.0, 4.0] {
            data.extend_from_slice(&x.to_le_bytes());
        }
        data.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        data.extend_from_slice(&half::f16::from_f32(-2.0).to_le_bytes());
        let header = r#"{"__metadata__":{"x":"y"},"embeddings":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"half":{"dtype":"F16","shape":[2],"data_offsets":[16,20]}}"#;
        let st = SafeTensors::from_bytes(file(header, &data), "t").expect("opens");
        assert_eq!(st.shape("embeddings"), Some(&[2, 2][..]));
        assert_eq!(
            st.tensor_f32("embeddings").expect("f32"),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(st.tensor_f32("half").expect("f16"), vec![0.5, -2.0]);
        assert!(st.tensor_f32("nope").is_err());
        let past = r#"{"e":{"dtype":"F32","shape":[2,2],"data_offsets":[0,32]}}"#;
        assert!(SafeTensors::from_bytes(file(past, &data), "t").is_err());
        let wrong = r#"{"e":{"dtype":"F32","shape":[3],"data_offsets":[0,16]}}"#;
        assert!(SafeTensors::from_bytes(file(wrong, &data), "t").is_err());
        assert!(matches!(
            SafeTensors::from_bytes(
                b"version https://git-lfs.github.com/spec/v1\n".to_vec(),
                "t"
            ),
            Err(InferError::LfsPointer { .. })
        ));
        assert!(SafeTensors::from_bytes(vec![1, 2, 3], "t").is_err());
    }
}
