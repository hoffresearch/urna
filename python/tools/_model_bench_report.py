"""The terminal report of urna_model_bench.py: one block per tier, never
one number across tiers (T1 is inflated by construction, T2 is a cost,
T3 is the utility ruler). Split out of urna_model_bench.py for the
300-line contract.
"""

from __future__ import annotations


def print_table(report: dict, ks: list[int]) -> None:
    print("\n== T1 pipeline stability (identity@k, inflated by construction) ==")
    for _preset, rep in report["models"].items():
        for name, sp in rep["spaces"].items():
            print(
                f"  {name:<24} "
                + "  ".join(f"id@{k}={sp['t1_identity_recall'][f'@{k}']:.3f}" for k in ks)
            )
    print("\n== T2 codec cost (drift cosine: source-embed vs stored decoded) ==")
    for _preset, rep in report["models"].items():
        for name, sp in rep["spaces"].items():
            d = sp["t2_drift_cosine"]
            print(f"  {name:<24} p10={d['p10']}  p50={d['p50']}")
    print("\n== T3 task utility (never compare against T1/T2 numbers) ==")
    for _preset, rep in report["models"].items():
        for name, sp in rep["spaces"].items():
            if "t3_text_to_image_hit" in sp:
                print(
                    f"  {name:<24} "
                    + "  ".join(f"txt@{k}={sp['t3_text_to_image_hit'][f'@{k}']:.3f}" for k in ks)
                    + f"  [{sp['t3_ruler']}]"
                )
    print("\n== cost ==")
    for _preset, rep in report["models"].items():
        for name, sp in rep["spaces"].items():
            mb = (sp.get("band_bytes") or 0) / 1e6
            print(
                f"  {name:<24} {mb:7.2f} MB  lat p50={sp['latency_ms']['p50']}ms "
                f"p95={sp['latency_ms']['p95']}ms  embed={rep['embed_side_items_per_s']} it/s"
            )
    if report.get("operator_queries"):
        print("\n== T3 operator queries ==")
        for name, r in report["operator_queries"].items():
            print(
                f"  {name:<24} "
                + "  ".join(f"hit@{k}={r['hit'][f'@{k}']}" for k in ks)
                + f"  mrr={r['mrr']} leak={r['negative_leakage']}"
            )
