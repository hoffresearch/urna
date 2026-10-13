# SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
#
# SPDX-License-Identifier: MIT

"""The validated installs of the presets setup offers, and the reviewed
code they run: data, kept apart from the registry (`presetmap`) that
attaches them.

an InstallSpec is recorded only after an install of exactly it, from an
empty hugging face cache, fingerprinted to its model_hash and answered a
query offline (`tool/tasks/presetjob.py` runs that for every preset when
the lock changes). a code pin is a file reviewed at the revision the
install pins; a changed file is refused, not warned about (N11).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstallSpec:
    """A validated install: the hub revision, the exact files fetched at it
    (path, bytes), the `model_hash` those files fingerprint to, and the pip
    specs its backend needs. a preset carries one only after an install of
    exactly this was run end to end and its hash checked; `catalogue`
    exposes only those. the file list is part of the identity: the
    fingerprint hashes every relevant file present, so fetching one more
    weights file (a pytorch_model.bin beside model.safetensors) would change
    the model_hash."""

    revision: str
    files: tuple[tuple[str, int], ...]
    model_hash: str
    packages: tuple[str, ...]


# the pip specs an install puts in the env, resolved together with every
# other offered model's into the lock (`catalogue.lock_inputs`).
_ST_PACKAGES = ("torch", "sentence-transformers>=5.7,<6", "transformers==5.2.0")
_JINA_TEXT_PACKAGES = _ST_PACKAGES + ("peft",)
_JINA_PACKAGES = _JINA_TEXT_PACKAGES + ("pillow",)
_WEMM_PACKAGES = _ST_PACKAGES + ("qwen-vl-utils==0.0.14", "pillow")
_OPEN_CLIP_PACKAGES = ("torch", "open_clip_torch", "pillow")
_SIGLIP_PACKAGES = _OPEN_CLIP_PACKAGES + ("transformers==5.2.0",)


# the sha256 of every python file a preset's repo runs, by preset.
CODE_PINS: dict[str, tuple[tuple[str, str], ...]] = {
    # pinned sha256 of the wemm-2b remote-code files as reviewed 2026-08-31
    # (N11): a changed file is REFUSED, not warned about. the same bytes are
    # at the revision its install pins (checked 2026-10-11).
    "wemm-2b": (
        ("modeling_st_wemm.py", "521d02c1c60ae727cc9dc6500cdb0b28c53b259e0ce3d37197920a33ba4dd333"),
        (
            "modeling_wemm_embedding.py",
            "ac255e1fad459cc3e68891d6c3327f4486922aed02fb3c5c13fb53277ba8e94f",
        ),
        ("chat_template.jinja", "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"),
        (
            "embedding_chat_template.jinja",
            "7c3df2aab83ab9096428ec27b6b99ad87c4790418b830d119957634f28c677ba",
        ),
        (
            "processor_config.json",
            "d601e2fe0de1bc11852de3aa843f01a1677ca84f4dc743916c9ce4b8d30fb384",
        ),
    ),
    # pinned sha256 of the python files sentence-transformers loads from the
    # jina v5 repos, reviewed 2026-10-11 at the revisions their installs pin.
    # each loads its adapters from the snapshot folder and opens no connection;
    # the omni ones fetch any http(s) string they are given as media, which the
    # st worker's offline guard refuses (stprocess.py). the vllm_*.py files serve
    # another runtime and are neither fetched nor run.
    "jina-v5-text-nano": (
        (
            "configuration_eurobert.py",
            "8737378a5cea9e6c7be0e077138d6c725fb5909ad122c979adcdcc1a005a51b3",
        ),
        (
            "configuration_jina_embeddings_v5.py",
            "8472183a99ba9767b1352309c7553b529738e9dcdc9fe703f3a130bb17d347cd",
        ),
        ("custom_st.py", "620865d4f54a9c45cf64d9ad15dc47825cd0c35686f7118a7d25f908e9fe563a"),
        (
            "modeling_eurobert.py",
            "f2430184a068bdfb5171650e8bbbff817aa5e7b2dd2066a52dc84d1f1d967f01",
        ),
        (
            "modeling_jina_embeddings_v5.py",
            "6d250e1abe0216b304b9ca99e49d9aac04e143671d310c112ad3e37b6186cf18",
        ),
    ),
    "jina-v5-omni-nano": (
        ("custom_st.py", "6b622359577d1bcef0658dd2a826ddb2aff3cffd1820a60454f4268be061d184"),
        (
            "modeling_jina_embeddings_v5_omni.py",
            "189d151cb05bc670a2170bf21c3bcb2cba7c201daca767940a516c1aa96f1679",
        ),
        (
            "modeling_llava_eurobert_audio.py",
            "9853d0126f175b0aca1463a57bad694b6b73c1c6eeea05d6cb8e73f2634f1c17",
        ),
        (
            "processing_llava_eurobert.py",
            "ce0cb53e8f9831a4aadf750580287adf4a4d2e73251c7fa789f03241d38311ad",
        ),
    ),
    "jina-v5-omni-small": (
        ("custom_st.py", "2be810bdca5dafe3553e656f88dba8b75b51e24e8173f0813950d5d530a7445e"),
        (
            "modeling_jina_embeddings_v5_omni.py",
            "bd6d6e35d6720beec881578b5db44f863ef72891732bb2fbd1da686c4e3ca07c",
        ),
    ),
}


INSTALLS: dict[str, InstallSpec] = {
    "minilm-multilingual": InstallSpec(
        revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        files=(
            ("1_Pooling/config.json", 190),
            ("README.md", 3888),
            ("config.json", 645),
            ("config_sentence_transformers.json", 122),
            ("model.safetensors", 470641600),
            ("modules.json", 229),
            ("sentence_bert_config.json", 53),
            ("special_tokens_map.json", 239),
            ("tokenizer.json", 9081518),
            ("tokenizer_config.json", 526),
        ),
        model_hash="sha256:97a500de1ad2dc9ffb41fbe3dec27d4fd26ddaa13143aa50f905d3d1f6eff0af",
        packages=("sentence-transformers>=3",),
    ),
    "harrier-270m": InstallSpec(
        revision="31de22b673913c7d658c0f03f792d77c2dcf8ebd",
        files=(
            ("1_Pooling/config.json", 296),
            ("README.md", 7606),
            ("config.json", 1326),
            ("config_sentence_transformers.json", 351),
            ("model.safetensors", 536221640),
            ("modules.json", 349),
            ("special_tokens_map.json", 662),
            ("tokenizer.json", 33385008),
            ("tokenizer_config.json", 1155377),
        ),
        model_hash="sha256:2b4de2399c6962589b6b700fc294c2ac9f8c45d535487652803d6a34ecaf7187",
        packages=_ST_PACKAGES,
    ),
    "jina-v5-text-nano": InstallSpec(
        revision="8a7f00aac812071b69403df470f1038ec85f8925",
        files=(
            ("README.md", 13968),
            ("adapters/classification/adapter_config.json", 883),
            ("adapters/classification/adapter_model.safetensors", 13587024),
            ("adapters/clustering/adapter_config.json", 882),
            ("adapters/clustering/adapter_model.safetensors", 13587024),
            ("adapters/retrieval/adapter_config.json", 882),
            ("adapters/retrieval/adapter_model.safetensors", 13586992),
            ("adapters/text-matching/adapter_config.json", 882),
            ("adapters/text-matching/adapter_model.safetensors", 13587024),
            ("config.json", 1127),
            ("config_sentence_transformers.json", 283),
            ("configuration_eurobert.py", 12138),
            ("configuration_jina_embeddings_v5.py", 137),
            ("custom_st.py", 4261),
            ("model.safetensors", 423543680),
            ("modeling_eurobert.py", 48939),
            ("modeling_jina_embeddings_v5.py", 4118),
            ("modules.json", 169),
            ("special_tokens_map.json", 582),
            ("tokenizer.json", 17210235),
            ("tokenizer_config.json", 50554),
        ),
        model_hash="sha256:b585d661980652d243344a2743cb5c07404a434831d1db9e88f7dcdd0232870c",
        packages=_JINA_TEXT_PACKAGES,
    ),
    "qwen3-embedding-0.6b": InstallSpec(
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        files=(
            ("1_Pooling/config.json", 313),
            ("README.md", 17237),
            ("config.json", 727),
            ("config_sentence_transformers.json", 215),
            ("model.safetensors", 1191586416),
            ("modules.json", 349),
            ("tokenizer.json", 11423705),
            ("tokenizer_config.json", 9706),
        ),
        model_hash="sha256:8201291e89d48850d4debe0238267161e42a8ef4738f388954694040d6b8f6fd",
        packages=_ST_PACKAGES,
    ),
    "octen-embedding-0.6b": InstallSpec(
        revision="b54faa2ba2ae9eb618f88526cec6dae3287eb5f8",
        files=(
            ("1_Pooling/config.json", 298),
            ("README.md", 7531),
            ("chat_template.jinja", 4116),
            ("config.json", 1354),
            ("config_sentence_transformers.json", 217),
            ("model.safetensors", 1191586416),
            ("modules.json", 350),
            ("special_tokens_map.json", 613),
            ("tokenizer.json", 11423705),
            ("tokenizer_config.json", 5404),
        ),
        model_hash="sha256:507aedde8cec86c8e05c74eb734285837531f18fd10799d92f153e1fc5a89d1e",
        packages=_ST_PACKAGES,
    ),
    "clip-vit-b32": InstallSpec(
        revision="a6f597a30f7b82c51704746581f9a4e41421e878",
        files=(
            ("open_clip_config.json", 556),
            ("open_clip_model.safetensors", 605143284),
        ),
        model_hash="sha256:76269ff6bb18be70a910506ccd31d52216b05a4bede0772ac17c0100ee75ca8f",
        packages=_OPEN_CLIP_PACKAGES,
    ),
    "siglip2": InstallSpec(
        revision="eee10eff6dd8cabae2d7f379d4e8cfcd352030aa",
        files=(
            ("open_clip_config.json", 944),
            ("open_clip_model.safetensors", 1500786832),
            ("special_tokens_map.json", 327),
            ("tokenizer.json", 34362885),
            ("tokenizer_config.json", 46382),
        ),
        model_hash="sha256:a9946db1d33627e773569c415ab202cd759befcdc047ac0d8768937298775f3c",
        packages=_SIGLIP_PACKAGES,
    ),
    "jina-v5-omni-nano": InstallSpec(
        revision="009e0d09a98a82227526e8c5aa7b0aa282e36163",
        files=(
            ("README.md", 17760),
            ("adapters/classification/adapter_config.json", 883),
            ("adapters/classification/adapter_model.safetensors", 13589512),
            ("adapters/clustering/adapter_config.json", 882),
            ("adapters/clustering/adapter_model.safetensors", 13589512),
            ("adapters/retrieval/adapter_config.json", 882),
            ("adapters/retrieval/adapter_model.safetensors", 13589512),
            ("adapters/text-matching/adapter_config.json", 882),
            ("adapters/text-matching/adapter_model.safetensors", 13589512),
            ("chat_template.jinja", 7755),
            ("config.json", 2415),
            ("config_sentence_transformers.json", 141),
            ("custom_st.py", 40952),
            ("model.safetensors", 1972058984),
            ("modeling_jina_embeddings_v5_omni.py", 27460),
            ("modeling_llava_eurobert_audio.py", 17033),
            ("modules.json", 104),
            ("processing_llava_eurobert.py", 7684),
            ("processor_config.json", 1447),
            ("tokenizer.json", 17211089),
            ("tokenizer_config.json", 624),
        ),
        model_hash="sha256:274d07bdbd4b8ff2d345667cd119e5dab26f015baed89891b137ea480d8b2280",
        packages=_JINA_PACKAGES,
    ),
    "jina-v5-omni-small": InstallSpec(
        revision="5c54692f22e186fc12ea38a9193e3ff9e1cda2a3",
        files=(
            ("README.md", 17723),
            ("adapters/classification/adapter_config.json", 884),
            ("adapters/classification/adapter_model.safetensors", 40426056),
            ("adapters/clustering/adapter_config.json", 883),
            ("adapters/clustering/adapter_model.safetensors", 40426056),
            ("adapters/retrieval/adapter_config.json", 883),
            ("adapters/retrieval/adapter_model.safetensors", 40426056),
            ("adapters/text-matching/adapter_config.json", 883),
            ("adapters/text-matching/adapter_model.safetensors", 40426056),
            ("chat_template.jinja", 7755),
            ("config.json", 2418),
            ("config_sentence_transformers.json", 141),
            ("custom_st.py", 40918),
            ("model.safetensors", 3252667112),
            ("modeling_jina_embeddings_v5_omni.py", 29076),
            ("modules.json", 105),
            ("processor_config.json", 1193),
            ("tokenizer.json", 11423488),
            ("tokenizer_config.json", 389),
        ),
        model_hash="sha256:4c7661eb37b0947b4e99847de04b1ce37caa51f176d377b5bc4508751b3a76e8",
        packages=_JINA_PACKAGES,
    ),
    "wemm-2b": InstallSpec(
        revision="bbd6cd4bf52cfc6716f752a2df80b2706720bd95",
        files=(
            ("README.md", 7993),
            ("additional_chat_templates/sentence_transformers.jinja", 1452),
            ("chat_template.jinja", 7755),
            ("config.json", 2853),
            ("config_sentence_transformers.json", 245),
            ("embedding_chat_template.jinja", 1086),
            ("model.safetensors", 5441695216),
            ("modeling_st_wemm.py", 3180),
            ("modeling_wemm_embedding.py", 1205),
            ("modules.json", 106),
            ("processor_config.json", 1192),
            ("sentence_bert_config.json", 950),
            ("tokenizer.json", 19990378),
            ("tokenizer_config.json", 1148),
        ),
        model_hash="sha256:acd71609394cdb51e55ee91953432bcde8b1e6c610dabe901a14e51465049c35",
        packages=_WEMM_PACKAGES,
    ),
}
