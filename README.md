# 🧬 GeneRx: Pharmacogenomic Data Fusion

**Preventing dangerous drug-gene and drug-food interactions at home.**

## 💡 The Elevator Pitch

Consumer LLMs shouldn't guess medical facts, and your raw DNA shouldn't be uploaded to the cloud. GeneRx is a privacy-first health dashboard that fuses a photo of your daily medications and food with your local DNA profile. By cross-referencing your unique genetic mutations against live FDA warnings, GeneRx alerts you to severe metabolic clashes before you take a pill.

## 🔗 The Hackathon Theme: FUSION

We built GeneRx to perfectly embody the **Fusion** theme by combining three completely different data modalities into one deterministic pipeline:

1. **The Physical World:** Unstructured photos of prescription bottles and dietary items (e.g., Grapefruit juice).

2. **Biological Data:** Raw, 600,000-line 23andMe DNA exports, filtered locally to find specific liver enzyme mutations (like CYP2C19 and CYP2D6).

3. **Live Regulatory Grounding:** Real-time data pulled from the U.S. `openFDA` API to guarantee clinical accuracy.

## 🏆 Best Use of Google Gemma 4

We didn't just use Gemma as a generic chatbot. We engineered a two-stage multimodal pipeline utilizing **Gemma 4 (27B)**:

* **Stage 1 (Vision Extraction):** Gemma 4 Vision acts as an OCR and categorization agent, reading curved pharmacy labels and identifying food items from a single user photo, outputting strict JSON.

* **Stage 2 (Clinical Reasoning):** Gemma 4 acts as the "Fusion Engine." It takes the parsed DNA mutations, the extracted visual data, and the dense medical text from the openFDA API, and deterministically reasons through the metabolic pathways to output a safe, plain-English medical verdict.

## ⚙️ How It Works

1. **Upload a Photo:** Snap a picture of your medication(s) and whatever you are eating/drinking.

2. **Local DNA Processing:** Upload your raw 23andMe `.txt` file. Our local Python filter scans 600,000+ rows, discards 99.9% of them for total privacy, and extracts only the 5 critical pharmacogenomic markers (SNPs).

3. **FDA API Grounding:** The backend queries the live `api.fda.gov` database for the detected drugs to fetch their official Boxed Warnings and Clinical Pharmacology.

4. **The Verdict:** Gemma 4 fuses this data to warn you if, for example, your specific `CYP2D6` mutation prevents you from safely metabolizing your medications.

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Python)

* **AI/LLM:** Google Gemma 4 (via OpenRouter API)

* **Data Sources:** U.S. openFDA API, 23andMe Raw Data format

* **Logic:** Python, Pillow