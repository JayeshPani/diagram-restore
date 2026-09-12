# Research search log

Date: 12 September 2026. Input: the user's supplied `pasted-text.txt` brief, read in full. Domain: black-and-white diagram restoration; proposed method: connectivity-aware diffusion routing; environment: one M4 Pro Mac. No deadline or fixed venue was supplied.

## Sources and coverage

Consensus paper discovery and fetched records; arXiv abstracts and full-text HTML; ICLR proceedings; CVF results; official author/project repositories; official PyTorch documentation; official conference sites. No geographic filter was used because this is a methods study. The Consensus query used a 2020–2026 filter; older foundational works were retained when directly relevant. Recent versions through the search date were checked where discovered.

## Exact discovery queries

Consensus:

```text
connectivity topology preserving diagram image restoration adaptive diffusion token routing year:2020-2026
```

Web — method/domain families:

```text
connectivity aware dynamic routing diffusion diagram restoration topology
topology preserving image restoration diffusion clDice diagram line drawing
adaptive computation image restoration token routing efficient restoration transformer
"topology" "diagram" "restoration" diffusion
"Region-Adaptive Sampling" diffusion
"clDice" topology CVPR 2021
"Denoising" "line drawings" restoration topology
"diagram" "restoration" "connectivity" deep learning
"flowchart" "dataset" "connectors" recognition official
"ARF: Arbitrary Routing Framework" doi
"Dynamic Diffusion Transformer" ICLR 2025 github
```

Web — venue feasibility:

```text
ICDAR 2027 official conference document analysis recognition
ICPR 2026 official conference paper deadline conference dates
```

Direct fetches covered the three supplied arXiv URLs, related papers/versions discovered through results and author repositories, MPS documentation, and the ICDAR 2027 call for papers. In-paper searches for `Spatial-wise Dynamic Token`, `router`, and `Training` located method and resource details. Searches for `clDice` in the DyDiT HTML and `topology` in DC-DiT HTML returned no match; absence of a keyword is **not** evidence of novelty.

## Consensus record handling

The initial query returned ten records. Fetched before use:

- ARF: record `f079be962ab25c019763577180372849`.
- TRaM-VSR: record `0618be65db3a53a1b2e210158ff798fa`, then cross-checked against arXiv:2607.22231.
- PixRestore: record `7d296b63a29d54e8a8127a14e3d36501`, then cross-checked against arXiv:2608.16793.

Primary arXiv metadata takes precedence over abbreviated/inconsistent author names in the discovery records. Unfetched Consensus hits are not cited as evidence.

## Access and version notes

- Palette, DyDiT, and DC-DiT: inspected relevant full-text HTML sections. DyDiT's ICLR 2025 status was verified in conference proceedings.
- DyDiT++: latest discovered v4 title and author-reported TPAMI acceptance checked; publisher bibliographic details remain unverified.
- TRaM-VSR and PixRestore: arXiv abstracts and metadata verified; HTML accessible, but this planning comparison deliberately limits their detailed claims to abstract evidence.
- clDice: arXiv abstract and author repository inspected. The CVF landing-page fetch failed; CVPR status is also identified by the author repository. No new theorem is inferred from the abstract.
- ARF: Consensus record fetched; DOI discovered in an indexed record. Publisher DOI fetch failed and PubMed returned a browser check. Treat this as abstract-level evidence, not a full-paper verification.
- RAS: official project site and `microsoft/RAS` inspected; an initially attempted longer repository name failed and is not treated as a separate project.
- PyTorch “stable” documentation redirected to version 2.14 during the search. The local package is 2.9.0; implementation must verify APIs against its locked version.
- The DyDiT family is reconciled as related versions; the newer Alibaba repository and original NUS repository are implementation references, not two independent experiments.

## Exclusions and unresolved leads

Excluded networking routing/topology repair, power-system restoration, geological restoration, and surface-rendering line drawings because their task is not diagram image restoration. Reddit, generic paper summaries, and third-party “papers with code” pages were used at most for discovery, not technical conclusions. Conference-name collisions were excluded in favor of the intended document-analysis/pattern-recognition organizations' sites.

Mural restoration, D²iT, ART, task-adaptive medical restoration, and flowchart-recognition datasets are adjacent leads. They were not promoted to detailed evidence in this initial review because exact direct-edge restoration/routing relevance or usable annotations/resources require further inspection. A public flowchart detector dataset is not automatically a licensed paired restoration benchmark.

## Limits and next update

The major concept families were covered, but recent searches were still revealing relevant work. This is therefore a bounded initial review, **not search saturation**. It supports a pilot and identifies novelty risks; it does not certify originality or exhaust the 2026 literature.

Before committing to large experiments: inspect the full recent methods, broaden graphics-recognition/diagram reconstruction coverage, inspect citations to the closest family, and verify implementation licenses at fixed revisions. Before submission: repeat the core queries, update arXiv versions, recheck venue dates/rules, and revise the contribution if new work overlaps.
