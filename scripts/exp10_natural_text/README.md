# Exp10: natural-text worlds (real Wikipedia documents with counterfactual renaming)

Design and registration: `/mnt/project-files/data/natural-text/DESIGN.md` (v1 + "v2 등록 변경" R1–R9), strict-reviewer
verdict `reviews/2026-10-16-natural-text-design-review.md` (CONDITIONAL GO). Paper wording: "a setting of real documents
with controlled counterfactual renaming", "multi-article" bundles.

## What a world is

Each world JSON in `data/natural_text/` mirrors the `gen_world.build_world` dict, so the exp07 ladder and the exp09 Qwen
port need only a loader switch:

| key | meaning |
|---|---|
| `triples` | one per target person: `id`, `pair`, fake `X` (full), `XF`, `XS`, birthplace `P`, in-bundle foil place `Q` (or null), school `E` (or null), `gender`, `natg` (name-language group), `region` (heuristic, may be null) |
| `texts` | `expo` units (`t`, `round`, `label` fact/name_only/value_only/other, `src` article/backlink, `n_fact_strict`, `contain8`, `text` with `@X{id}@` / `@XF{id}@` / `@XS{id}@`) and `bg` units (unrelated Wikipedia paragraphs) |
| `ctrl_persons`, `ctrl_places`, `pseudo_persons`, `pseudo_places` | per-triple never-seen controls, token-length matched |
| `frames` | held-out probe frames (birth, school, reverse, familiarity) + 4 EM frames |
| `stats` | fact / name exposure counts per person, fact repeats inside 8K, exposure counts before and after the deletion rules |
| `layout` | `spread` (main: rounds, pair partners adjacent), `clustered` (registered auxiliary: each bundle contiguous), `factonly` (control: name-only units replaced by same-length background) |

No real name, wiki title or Wikidata id is in any world file (`checks.py` asserts it). The real-name map lives only in
`/mnt/project-files/data/natural-text/private/` and must never be committed (`.gitignore`).

`nat_world.py` gives `realize(world, swap)`, `probes(world, name)`, `build_items(world)` (families L3, XP, PS, FN, FP,
FN0, FP0, L3r, L2 (empty), then L3b and L3e) with candidates fixed across worlds A/B, `forbid_strings(world)` for the
readout tails, and `l1_rows_text(world, swap, 'doc'|'fact')` for the L1 positive control. World B swaps the three name
slots inside each pair; every other character is identical (checked).

## Rebuild (CPU, about 25 min, deterministic)

```
export NT_DIR=/some/dir   # needs dl/: wikimedia/wikipedia 20231101.en parquet w0..w40, PopQA, T-REx (relbert/t_rex train),
                          # llama3 tokenizer (NousResearch/Meta-Llama-3-8B), Qwen3-4B-Base tokenizer, squad_dev.json
python3 -I scan1.py && python3 -I merge1.py && python3 -I bundles.py && python3 -I names.py && python3 -I scan2.py
python3 -I build.py && python3 -I checks.py $NT_DIR/out $NT_DIR/work/persons.pkl $NT_DIR/dl/squad_dev.json
```

## Still to do on GPU (10/19–24, before the streams are frozen)

1. `identity_screen.py` on Qwen3-4B-Base and the same score on 3B W0 (JAX, lab server); exclude a pair if either member
   ranks its real surname in the top 10 of ~1,000 surnames in either model. Needs the private map, copied by hand.
2. W0 margin screen: drop pairs whose |m| on L3 exceeds the top 10% of the synthetic W0 distribution; report m_A, m_B.
3. Rebuild the final streams from the surviving pairs (build.py takes the pair list from the pool; reserve streams 9005,
   9006 absorb drops). The R1 tail (pg1342) check for world strings runs where that file lives (`world_text.build_items`
   already asserts names are absent; extend it to `nat_world.forbid_strings`).
