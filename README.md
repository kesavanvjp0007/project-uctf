# Project UCTF
## An Open Research Program on Machine-Native AI Training Representations

**Author:** K7007  
**Status:** Papers 1 & 2 Complete — Paper 3 In Design  
**Started:** July 2026  
**HuggingFace:** [Project UCTF Discussion](https://discuss.huggingface.co/t/project-uctf-an-open-research-program-on-machine-native-ai-training-representations/177240)

---

## Core Research Question

Can a machine-native intermediate representation replace raw multilingual text as an AI training format — reducing cross-lingual redundancy while preserving the information required for downstream generation?

---

## Research Roadmap

| Paper | Title | Status |
|-------|-------|--------|
| Paper 1 | Measuring Semantic Redundancy in Multilingual Training Corpora | ✅ Complete |
| Paper 2 | Characterizing Universal and Language-Specific Knowledge | ✅ Complete |
| Paper 3 | Design Requirements for a Machine-Native Intermediate Representation | 🔄 In Design |
| Paper 4 | Building the UCTF Encoder | ⏳ Planned |
| Paper 5 | First UCTF Training Experiment | ⏳ Planned |

Each paper has standalone value. If any stage fails, prior stages remain valid contributions.

---

## Paper 1 — Measuring Semantic Redundancy in Multilingual Training Corpora

**Research question:** How much semantic duplication exists across languages in real multilingual AI training datasets?

### 📄 PDF Report
[UCTF_Paper1_Final_Summary.pdf](paper1_final/UCTF_Paper1_Final_Summary.pdf)

### Experiments

| Version | Scale | Mean Similarity | Key Finding |
|---------|-------|----------------|-------------|
| v0 | 20 pairs manual | 0.9570 | 100% precision at all thresholds |
| v1 | 1,500 real pairs | 0.7474 | Real data noisier — Eng-Jap gap 0.136 |
| v2 | 4 encoders GPU | 0.8820 (mE5) | mE5 best — zero negatives, std 0.046 |
| v3a | 900 noisy sentences | 0.9627 avg | mE5 robust — 98.7–100% above 0.85 |
| v3b | 600 low-resource | 0.9045 avg | Sinhala, Nepali match high-resource |
| v4 | 10,000 pairs corpus | 0.8948 | 1,515 sent/sec · 458MB VRAM · RTX 3050 |

### Key Findings
- Text deduplication detects **0%** cross-lingual redundancy — embedding approaches are necessary
- **mE5-small** is the strongest encoder — mean 0.8820, std 0.0455, zero negative pairs
- mE5 maintains **94.9–97.5%** similarity under character noise, word shuffling, and code-switching
- Low-resource languages (Sinhala 91.8%, Nepali 90.4%) outperform Chinese (87.6%) and Japanese (86.7%) at corpus scale — reason under investigation

### Files

| File | Description |
|------|-------------|
| [paper1_v0/paper1_v0.py](paper1_v0/paper1_v0.py) | v0 experiment code |
| [paper1_v0/flores_4lang.csv](paper1_v0/flores_4lang.csv) | Aligned 4-language dataset |
| [paper1_v0/candidate_pairs.csv](paper1_v0/candidate_pairs.csv) | 116 semantic candidate pairs |
| [paper1_v0/threshold_summary.csv](paper1_v0/threshold_summary.csv) | Precision at 7 thresholds |
| [paper1_v0/manual_audit_template.csv](paper1_v0/manual_audit_template.csv) | Human audit template |
| [paper1_v0/uctf_paper1_results.png](paper1_v0/uctf_paper1_results.png) | v0 visualization |
| [paper1_v1/flores_v1.csv](paper1_v1/flores_v1.csv) | 1,500 real pairs dataset |
| [paper1_v1/v1_similarity_results.csv](paper1_v1/v1_similarity_results.csv) | v1 similarity results |
| [paper1_v2/v2_encoder_comparison.csv](paper1_v2/v2_encoder_comparison.csv) | 4-encoder comparison (6,000 records) |
| [paper1_v3/v3_noise_results.csv](paper1_v3/v3_noise_results.csv) | Noise robustness results |
| [paper1_v3/v3_lowresource_results.csv](paper1_v3/v3_lowresource_results.csv) | Low-resource language results |
| [paper1_v4/v4_results.csv](paper1_v4/v4_results.csv) | Corpus scale results (10,000 pairs) |
| [paper1_final/UCTF_Paper1_Final_Summary.pdf](paper1_final/UCTF_Paper1_Final_Summary.pdf) | Full PDF report |

### How to Reproduce

```bash
git clone https://github.com/kesavanvjp0007/project-uctf
cd project-uctf
python -m venv venv
venv\Scripts\activate
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install sentence-transformers==3.0.1 transformers==4.44.0
pip install datasets pandas numpy matplotlib seaborn scikit-learn
cd paper1_v0
python paper1_v0.py
```

**Environment:** Python 3.12 · PyTorch 2.5.1+cu124 · sentence-transformers 3.0.1 · transformers 4.44.0 · CUDA 12.4 · NVIDIA RTX 3050 4GB

---

## Paper 2 — Characterizing Universal and Language-Specific Knowledge

**Research question:** Which semantic information is universal and safely compressible — and which is language-dependent and must be preserved?

### 📄 PDF Report
[UCTF_Paper2_Final_Summary.pdf](paper2/UCTF_Paper2_Final_Summary.pdf)

### Dataset
60 sentences × 6 knowledge categories × 4 languages = 240 sentences  
Categories: `factual` · `mathematical` · `named_entity` · `emotional_nuance` · `cultural_idiom` · `grammar_dependent`

### Results

| Category | Mean | Std | Above 0.90 | UCTF Safety |
|----------|------|-----|------------|-------------|
| grammar_dependent | 0.9148 | 0.0213 | 76.7% | ✅ Safe |
| emotional_nuance | 0.9069 | 0.0244 | 60.0% | ✅ Safe |
| factual | 0.8968 | 0.0216 | 46.7% | ✅ Safe |
| mathematical | 0.8795 | 0.0309 | 25.0% | ⚠️ Medium |
| named_entity | 0.8749 | 0.0318 | 16.7% | ⚠️ Medium |
| cultural_idiom | 0.8573 | 0.0470 | 20.0% | 🔴 Caution |

### 3-Tier UCTF Compression Strategy

| Tier | Categories | Preservation | Strategy |
|------|-----------|-------------|----------|
| Tier 1 — Safe | factual, mathematical, grammar, emotion | 88–91% | Full UCTF compression |
| Tier 2 — Flagged | named entities | 87.5% | Compress + preserve proper nouns separately |
| Tier 3 — Preserve | cultural idioms | 85.7% | Store as language-specific tokens |

### Key Findings
- Hypothesis partially reversed — grammar-dependent scored **highest** (91.5%), not factual or mathematical
- Cultural idioms are the **primary compression risk** — lowest preservation (85.7%), highest variance (std 0.0470), only 58.3% above 0.85 threshold
- Emotional concepts are **more cross-linguistically universal** than initially assumed (90.7%)
- Named entities show moderate divergence — cultural references diverge more than scientific proper nouns
- Most universal sentence: *"She gave him a gift yesterday."* — sim=0.9459
- Most divergent sentence: *"It is raining cats and dogs."* — sim=0.8103

### Files

| File | Description |
|------|-------------|
| [paper2/paper2_dataset.csv](paper2/paper2_dataset.csv) | 240-sentence labeled dataset |
| [paper2/paper2_results.csv](paper2/paper2_results.csv) | Category similarity results |
| [paper2/paper2_sentence_analysis.csv](paper2/paper2_sentence_analysis.csv) | Per-sentence analysis |
| [paper2/paper2_results.png](paper2/paper2_results.png) | Category visualization |
| [paper2/paper2_compression_analysis.png](paper2/paper2_compression_analysis.png) | Compression analysis charts |
| [paper2/UCTF_Paper2_Final_Summary.pdf](paper2/UCTF_Paper2_Final_Summary.pdf) | Full PDF report |

---

## Environment

```
Python:                3.12.0
PyTorch:               2.5.1+cu124
sentence-transformers: 3.0.1
transformers:          4.44.0
datasets:              5.0.0
CUDA:                  12.4
GPU:                   NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
OS:                    Windows 11
```

---

## Research Philosophy

- Every experiment answers one question
- Conclusions follow evidence
- Negative results are published
- All code and data released publicly
- Community feedback incorporated at every stage
- Credit given explicitly wherever ideas came from others

---

## Limitations

**Paper 1:** OPUS-100 is a parallel corpus — cleaner than real web-crawled training data. Only 7 languages tested. Word-shuffle noise may indicate encoder insensitivity to word order rather than real robustness (flagged by community reviewer).

**Paper 2:** 10 sentences per category is a small sample. Only 4 languages tested. mE5-small used — larger encoders may show different results. Preservation estimates are proxy measures — actual UCTF information loss requires encoder-decoder experiments in Paper 4.

---

## How to Contribute

- Literature recommendations
- Dataset suggestions
- Methodology critique
- Experiment replication
- Identifying prior work or flaws
- Constructive criticism at any stage

Finding weaknesses early is as valuable as proposing improvements.

---

## Acknowledgements

Special thanks to **@John6666** on HuggingFace Forums whose detailed technical feedback directly shaped the research architecture of this program and whose checkpoint review improved the quality of Paper 1 significantly.

---

*The goal is not to prove UCTF is possible. The goal is to find out.*

— K7007
