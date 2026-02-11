# Chemistry Content Crew — CrewAI + Ollama (local) Integration

## Overview

`Chemistry Content Crew` is an automated scientific content generation workflow built with CrewAI and local LLMs served via Ollama, integrated through an OpenAI-compatible interface. The system aims to produce scientifically rigorous, pedagogically clear, and SEO-optimized chemistry articles, operating fully offline and without relying on paid APIs. The architecture mirrors a real editorial pipeline by separating responsibilities among specialized agents.

## Technology Rationale

- **Why CrewAI?** Enables multi-agent orchestration that reflects a real editorial workflow: content strategy, scientific review, didactic writing, and technical/SEO auditing. Each agent has clearly defined roles and boundaries, reducing conceptual drift and hallucinations.
- **Why Ollama (Local LLMs)?** 100% local execution (no per-token costs), full control over models and versions, and stronger data privacy — suitable for frequent long-form generation.
- **Why emulate the OpenAI API?** Many libraries (including CrewAI) use the OpenAI client; by pointing `OPENAI_API_BASE` to the local Ollama server, calls can be routed locally without changing library internals.

## Features

- Multi-agent orchestration for scientific content.
- Local LLM inference via Ollama.
- Clear separation of responsibilities (Strategy, Rigor, Didactics, SEO).
- Automatic Markdown article generation.
- Automatic persistence of results to disk.

## Quick Start

1. Clone and install dependencies

```bash
git clone <your-repository>
cd <your-repository>
pip install -r requirements.txt
```

2. Download/prepare models in Ollama

```bash
ollama pull llama3.2
ollama pull phi3.5
ollama pull qwen2.5:3b
```

3. Run the workflow

```bash
python main.py
```

The final article will be generated and saved automatically.

## Prerequisites

- Python 3.9+
- Ollama installed and serving (e.g., `ollama serve`)
- Recommended models: `llama3.2`, `phi3.5`, `qwen2.5:3b` (verify names/availability in your Ollama registry)
- Core libraries: `crewai`, `openai`, `langchain`

## OpenAI → Ollama Local Integration (example)

Before importing `openai` or `crewai`, configure the environment to redirect calls:

```python
import os
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:11434/v1")
os.environ.setdefault("OPENAI_API_KEY", "ollama")
```

Ensure the Ollama service is running and responding on the specified port.

## LLM Configuration Strategy

Each model is selected according to its cognitive role in the pipeline:

- `llama3.2` — Didactic writing: long-form coherence and clarity.
- `phi3.5` — Scientific rigor: precision and lower hallucination risk.
- `qwen2.5:3b` — Strategy & SEO: structural analysis and analytical tasks.

Example (pseudo-code) configuration using an LLM abstraction:

```python
model_content = LLM(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434"
)
```

## Agent Architecture

1. **Scientific Strategy Agent**
   - Responsibilities: topic selection, SEO research, search intent analysis, and article structure (H1–H3).
   - Constraints: does not write the final content; only defines the editorial blueprint.

2. **Chemistry Subject Matter Expert (SME)**
   - Responsibilities: formal definitions, chemical laws, equations, and reaction mechanisms.
   - Constraints: focuses exclusively on scientific rigor; does not perform didactic adaptation.

3. **Didactic Chemistry Educator**
   - Responsibilities: pedagogical translation, real-world applications, and accessible explanations.
   - Constraints: does not introduce new scientific facts; works from the SME dossier.

4. **Technical Editor & SEO Guardian**
   - Responsibilities: final audit, verification of formulas/units, SEO optimization, and Markdown consistency.
   - Constraints: never alters scientific truth for SEO purposes.

## Task Pipeline

- Strategy Task: defines topic, audience, keywords, and structure.
- Expert Science Task: produces a technical scientific dossier.
- Didactic Writing Task: drafts the full article following the blueprint.
- Final Optimization Task: scientific audit and SEO refinement.

## Output and Persistence

Results are saved automatically as Markdown:

- `resultado.md` — Portuguese version
- `result.md` — English version

## Known Limitations

- Performance depends on local hardware (GPU vs CPU). Models benefit greatly from GPU acceleration.
- No persistent memory between separate executions unless additional storage is implemented.
- SEO insights are inference-based (offline) and may differ from live online tools.

## Project Structure

```
project/
├── main.py            # Crew orchestration and execution
├── resultado.md       # Final article (PT)
├── result.md          # Final article (EN)
├── requirements.txt   # Dependencies
└── README.md          # Documentation
```

## Troubleshooting

- **Ollama not responding**: ensure the service is running (`ollama serve`).
- **Model not found**: run `ollama pull <model_name>` and verify model names.
- **Scientific inconsistencies**: reduce the temperature of the Expert agent and verify sources/references.

## Acknowledgments

- CrewAI for multi-agent orchestration.
- Ollama for local LLM inference.
- LangChain for model abstraction.

---

## Technical Assessment (summary)

- The core ideas are correct and plausible: using a local Ollama instance and pointing `OPENAI_API_BASE` to it is a common approach to emulate the OpenAI API.
- **Validations required:**
  - Confirm that the model identifiers (`llama3.2`, `phi3.5`, `qwen2.5:3b`) exist in your Ollama registry — names and availability vary by provider and license.
  - Verify compatibility of `crewai`/`langchain` versions with custom OpenAI endpoints.
  - Test latency and memory usage; larger models may be impractical on CPU-only systems.
- **Recommended best practices:**
  - Pin exact dependency versions in `requirements.txt`.
  - Add sanity checks in the workflow (validate model responses, verify units/formulas).
  - Include bibliographic references and external verification for critical claims.
- **Risks/notes:**
  - Some models may have licensing restrictions for offline or commercial use; verify terms.
  - Lowering the temperature reduces hallucinations but does not eliminate factual errors — human expert review is advised.

If you want, I can run automatic validations (check `ollama list`, test the local OpenAI-compatible endpoint, and inspect `requirements.txt`).
