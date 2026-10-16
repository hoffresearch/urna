//! What the manifest says about a corpus beyond its vectors: the meaning
//! of each chunk_scalars (0x0D) column, where the data came from, and how
//! the stored media derives from the originals. All of it is additive and
//! optional, serialized only when set, so a file without it stays
//! byte-identical to a v1 manifest.

use serde::{Deserialize, Serialize};

/// One column of chunk_scalars (0x0D), as an app reads it: `kind` is the
/// column's wire kind name (`int`, `real`, `bool`, `text`, `uuid`, `enum`,
/// `enum_list`), `roles` say what the app does with it (`filter`, `facet`,
/// `label`, `display`), `prefix` is the search-syntax key (`t` for a type
/// line, `c` for colors), and `display` how a value is drawn (`text`,
/// `chip`, `symbols`, `number`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SchemaField {
    pub name: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roles: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
}

impl SchemaField {
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }
}

/// Where the corpus data came from. `sha256` is the hash of the bulk file
/// the build read, so a rebuild can prove it read the same bytes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DatasetSource {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// ISO-8601 date the data was obtained.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fetched: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

/// How the stored media derives from its originals: the blob_refs (0x14)
/// `content_hash` of each record is the sha-256 of the ORIGINAL bytes, and
/// this says which codec, profile and quality produced what is stored.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MediaDerivation {
    pub codec: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lossy: Option<bool>,
}
