# EVE Online Chat Topic Classifier

This repository provides a pipeline for classifying chat messages from EVE Online into game-relevant topics using a Large Language Model (LLM) via LangChain and Gemini. The main innovation is the addition of game-specific context to the prompt and a redesigned LLM reasoning pipeline, which together achieve improved classification accuracy over the original approach.

## Project Highlights

- **Game-Specific Context:** The classifier injects detailed EVE Online context into the LLM prompt, enabling more accurate and relevant topic identification.
- **Redesigned LLM Pipeline:** The reasoning and classification steps are restructured to leverage the LLM's strengths, resulting in better performance on annotated chat data.

## Prerequisites

- Python 3.13.3 is tested.
- Data files (should be present in the repo root):
  - `eve_crash_classified - eve_crash_classified.csv` — annotated chat messages
  - `topics.csv` — topic definitions

## Installation

1. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

- Obtain API access for Google Generative AI (Gemini).
- Create a `.env` file in the repo root with your API key and any required environment variables. See `.env.example` for reference.

## Usage

Run the classifier with:

```bash
python classifier.py
```

- The script loads annotated chat data and topic definitions.
- It applies game-specific context to the LLM prompt.
- The redesigned pipeline classifies messages into topics, refines results based on confidence, and writes improved classifications to `improved.csv`.

## Output

- `improved.csv`: Contains the corrected and improved topic classifications for the processed chat messages.

## Troubleshooting & Notes

- Ensure your `.env` file is correctly configured with valid API keys.
- All dependencies must be installed (see `requirements.txt`).
- Data files must be present in the repo root.
- If you encounter issues with LLM access, check your API credentials and network connectivity.

## Example Workflow

1. Ensure `eve_crash_classified - eve_crash_classified.csv` and `topics.csv` are present.
2. Install dependencies: `pip install -r requirements.txt`
3. Set up `.env` with your Gemini API key.
4. Run: `python classifier.py`
5. Review results in `improved.csv`.

## Innovation

This project improves upon standard topic classification by:
- **Injecting game-specific context** into the LLM prompt, allowing the model to understand nuanced, domain-specific language and references.
- **Redesigning the LLM reasoning pipeline** to better utilize structured outputs and confidence thresholds, resulting in more accurate and actionable classifications.
