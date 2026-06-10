# Release & Publication Checklist

Tracks the remaining steps to take Reson from "professional repo" to
"citable, published work." Items marked (manual) require account
access or physical action and cannot be automated from the repo.

## Done

- [x] MIT LICENSE
- [x] CI (GitHub Actions, pytest on Python 3.11–3.13)
- [x] CITATION.cff (GitHub "Cite this repository" button)
- [x] .zenodo.json (Zenodo release metadata)
- [x] CONTRIBUTING.md
- [x] pyproject metadata (authors, license, classifiers, URLs)
- [x] README badges, citation, and license sections
- [x] v0.1.0 tag/release

## Data & validation (manual — physical)

- [ ] Record 5–10 prompted sessions per `docs/data_collection_protocol.md`,
      with reattachment between some sessions, rest-only segments, and
      artifact-only segments (see `docs/validation_status.md`).
- [ ] Reserve held-out sessions before any model comparison.
- [x] Baseline comparison tooling: `reson-eval` runs leave-one-session-out
      WL-threshold / WL-logreg / all-feature-logreg and reports event-level
      metrics. (Run it once the re-recorded + new sessions are in.)
- [ ] Write results into `docs/validation_status.md` and the paper.

## Publication (manual — accounts)

- [x] GitHub account renamed `JJ9276489` → `jeraldhu-yuan` (2026-06-10).
      Still to do: fill in profile (display name, affiliation, photo).
- [x] Zenodo: linked, v0.2.1 archived, DOI 10.5281/zenodo.20631777 minted
      (concept DOI). Badge added to README and `doi` added to CITATION.cff.
- [ ] Dataset: package the validation sessions (raw.csv, features.csv,
      labels.jsonl, meta.json per session) and upload to Zenodo as a
      separate dataset deposit with its own DOI.
- [x] Demo video recorded and linked from README:
      https://www.youtube.com/watch?v=78xajhyyDSc
      Optional next: also archive it to Zenodo for a citable video DOI.
- [ ] JOSS: finish `paper/paper.md` TODOs after validation, then submit
      at https://joss.theoj.org (requirements: OSI license ✓, tests ✓,
      docs ✓, contribution guidelines ✓).
- [ ] Optional preprint: system + validation write-up on arXiv
      (cs.HC) or TechRxiv for an immediately citable record.

## Community seeding (manual)

- [ ] Post to assistive-technology communities (r/assistivetechnology,
      AAC/ALS forums, Hacker News "Show HN").
- [ ] Identify 1–2 assistive-technology researchers/clinicians to demo
      the project to; ask for feedback (and, later, letters).
