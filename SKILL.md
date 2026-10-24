---
name: generx-pharmacogenomic-fusion
description: >-
  Use when asked to interpret pharmacogenomic or pharmacogenetic data, scan a
  23andMe/AncestryDNA raw-genotype file for clinically actionable CYP / VKORC1 /
  SLCO1B1 / CYP2D6 / CYP2C19 variants, fuse genetic insight with a drug
  prescription and lifestyle factors, or evaluate whether a medication dose
  should be adjusted. Also use when reviewing the GeneRx fusion pipeline, the
  offline DNA privacy filter, or the live openFDA evidence layer. This skill
  treats medication decisions as an auditable, three-modality evidence problem
  and refuses to fabricate clinical conclusions.
---

# GeneRx — Pharmacogenomic & Environmental Data Fusion

## What this skill is

GeneRx is a hackathon-grade clinical decision-support prototype that fuses
**three independent evidence streams** into one pharmacogenomic verdict:

1. **Vision modality** — a photo of the patient's prescription / medication
   packaging is read by a multimodal LLM (Gemma 4 via OpenRouter) and
   returns a strict JSON extraction: candidate `drugs` and
   `dietary_and_otc_items`.
2. **DNA modality (privacy-first, local)** — a raw 23andMe-style genotype file is
   scanned by `clinical_data.py`; only clinically relevant SNPs are retained and
   the rest are discarded locally (the tool reports its local privacy
   ("discard") rate).
3. **Live pharmacology modality** — the generic drug name is looked up against
   the live **openFDA** drug-label API, falling back to verbatim, spliced
   excerpts from real FDA labels.

Those three outputs are fused by a fourth LLM pass (`evaluate_fusion_gemma`)
which produces a risk level (`CRITICAL` / `MODERATE` / `SAFE`), a headline, the
metabolic pathway status, mechanism of interaction, environmental clash,
FDA-verbatim citation, and a clinical recommendation.

## When to use this skill

- The user asks to analyze genetic/pharmacogenomic data for a medication.
- The user has a 23andMe-style file or a mock genotype and wants a risk verdict.
- The user wants to know how a drug interacts with genes + diet/OTC.
- The user wants to verify or audit the GeneRx pipeline code.

## How to use the skill

1. **Scan DNA**: call `pipeline.parse_23andme_file(file_text)` — it returns
   `matched_snps`, `total_snps_scanned`, and `privacy_discard_rate`.
2. **Interpret**: use `pipeline.relevant_snps(profile_type)` and
   `pipeline.interpret_genotype(...)` to map genotype → phenotype.
3. **Fetch label**: call `pipeline.fetch_openfda(generic_name)` — returns
   `{"source": ..., "data": ...}` with `source` being `"live_openfda"` or
   `"fallback_fda"`.
4. **Extract from images**: call `pipeline.extract_vision_gemma(image_bytes, api_key, model_name, base_url)` per photo, then merge the JSON results (medication photos yield drugs, food/dietary photos yield co-ingestants). `pipeline.run_fusion` accepts a single image or a list of images.
5. **Fuse**: call `pipeline.evaluate_fusion_gemma(...)` for the final verdict.

## Hard rules for agents

- Never fabricate a clinical recommendation or a quoted FDA sentence that the
  model did not produce. Use `pipeline.verify_citation(...)` and substitute the
  verified `exact_text` before showing anything to a user.
- If fewer than two modalities are available (e.g., no API key, no image, no
  DNA file), surface the missing modality instead of guessing.
- Local endpoints (e.g. an Ollama server) require no API key; the sidebar
  auto-selects the best local base URL/model when no key is set.
- Respect the privacy posture: raw genetic data never leaves the machine unless
  a real vision API call is made with an explicit key.

## Reference implementations

- `clinical_data.py` — SNPs, synthetic genotype generator, mock label images.
- `pipeline.py` — the fusion pipeline (parsing, openFDA, Gemma vision, fusion).
- `app.py` — Streamlit UI (`streamlit run app.py`).
