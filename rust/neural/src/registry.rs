//! Which graph and tokenizer a corpus's `embedding_model` needs. A model
//! enters here when its graph ships; an unknown name gets `None`, and the
//! caller falls back to whatever it did before.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Graph {
    /// a static embedding table: gather, mean pool, l2 normalize.
    Static,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerKind {
    WordPiece,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelSpec {
    pub graph: Graph,
    pub tokenizer: TokenizerKind,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Registry;

impl Registry {
    /// The spec for `embedding_model` as the manifest names it.
    pub fn for_model(embedding_model: &str) -> Option<ModelSpec> {
        match embedding_model {
            "minishlab/potion-base-8M" => Some(ModelSpec {
                graph: Graph::Static,
                tokenizer: TokenizerKind::WordPiece,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_potion_is_registered_and_strangers_are_not() {
        let spec = Registry::for_model("minishlab/potion-base-8M").expect("potion");
        assert_eq!(spec.graph, Graph::Static);
        assert_eq!(spec.tokenizer, TokenizerKind::WordPiece);
        assert_eq!(Registry::for_model("org/other"), None);
    }
}
