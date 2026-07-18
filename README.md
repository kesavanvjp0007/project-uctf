# Project UCTF
## An Open Research Program on Machine-Native AI Training Representations

**Author:** K7007  
**Status:** Active — Paper 1 v0 Complete  
**Started:** July 2026  
**HuggingFace Discussion:** [Project UCTF on HF Forums](https://discuss.huggingface.co)

---

## Core Research Question

Can a machine-native intermediate representation replace raw multilingual text
as an AI training format — reducing cross-lingual redundancy while preserving
the information required for downstream generation?

This is a hypothesis. Not a conclusion.

---

## The Problem

The same semantic concept expressed across 100 languages is stored 100 times
in current AI training datasets. From a machine learning perspective, this is
the same information stored redundantly.

Combined with the projected exhaustion of usable human-generated training data
between 2026 and 2032, this redundancy represents a structural inefficiency
the field has not fully addressed.

---

## Paper 1 — Measuring Semantic Redundancy in Multilingual Training Corpora

### Status: v0 Complete ✅

### Research Question
How much semantic duplication actually exists across languages
in real training datasets?

### v0 Results

| Metric | Result |
|--------|--------|
| Dataset | 20 sentences × 4 languages = 80 total |
| Languages | English, Chinese, Japanese, Russian |
| Encoder | paraphrase-multilingual-MiniLM-L12-v2 |
| Exact duplicates | 0 (0.0%) |
| Semantic pairs at threshold 0.85 | 116 of 120 possible |
| Precision across all thresholds | 100% |
| Average cross-language similarity | 95.70% |
| Min similarity | 85.29% |
| Max similarity | 99.33% |
| False positives | 0 |

### Key Finding
Exact duplicate detection fails completely at capturing cross-lingual
semantic redundancy — 0% detection rate. Embedding-based semantic
detection achieved 100% precision across all tested thresholds (0.70–0.99),
with average cross-language similarity of 95.70%.

This confirms that multilingual training datasets contain significant
measurable semantic redundancy that text-level deduplication cannot detect.

### Files

paper1_v0/
├── paper1_v0.py              — full experiment code
├── flores_4lang.csv          — aligned 4-language dataset
├── candidate_pairs.csv       — 116 semantic candidate pairs
├── threshold_summary.csv     — precision at 7 thresholds
├── exact_duplicate_summary.json  — baseline results
├── manual_audit_template.csv — human audit template
├── embeddings.npy            — sentence embeddings
└── uctf_paper1_results.png   — result visualizations


---

## Full Research Roadmap

| Paper | Title | Status |
|-------|-------|--------|
| Paper 1 | Measuring Semantic Redundancy in Multilingual Training Corpora | v0 Complete ✅ |
| Paper 2 | Characterizing Universal and Language-Specific Knowledge | Planned |
| Paper 3 | Design Requirements for a Machine-Native Intermediate Representation | Planned |
| Paper 4 | Building the UCTF Encoder | Planned |
| Paper 5 | First UCTF Training Experiment | Planned |

Each paper has standalone value.
If any stage fails, prior stages remain valid contributions.

---

## Research Philosophy

- Every experiment answers one question
- Conclusions follow evidence
- Negative results are published
- All code and findings released publicly
- Community feedback incorporated at every stage

---

## How to Run Paper 1 v0

```bash
# Clone the repository
git clone https://github.com/kesavanvjp0007/project-uctf

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install sentence-transformers datasets pandas numpy matplotlib seaborn scikit-learn

# Run experiment
cd paper1_v0
python paper1_v0.py
```

---

## How to Contribute

- Literature recommendations
- Dataset suggestions  
- Methodology critique
- Experiment replication
- Identifying prior work
- Identifying flaws before implementation

Finding weaknesses early is as valuable as proposing improvements.

---

## Concept Paper

Full concept paper available in `/docs/UCTF_Concept_Paper_2026.docx`

---

## Acknowledgements

Special thanks to **@John6666** on HuggingFace Forums whose detailed
technical feedback directly shaped the research architecture of this program.

---

## License

All research outputs in this repository are released openly for academic
and research use. Please cite Project UCTF and link to this repository
if building on this work.

---

*The goal is not to prove UCTF is possible. The goal is to find out.*

— K7007