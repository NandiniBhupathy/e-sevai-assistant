# esevai assistant

Tamil-English retrieval-augmented QA chatbot over Tamil Nadu e-Sevai operator training manuals.

## What it does

- Accepts English, Tamil, and mixed questions.
- Routes service-specific questions to the relevant manual.
- Uses PDF text plus OCR from rendered screenshot crops.
- Retrieves with BGE-M3 + FAISS and BM25, fused with reciprocal rank fusion.
- Answers in the user's language and cites manual/page.
- Shows at most two screenshot crops actually used by the answer.
- Abstains when the manual does not cover the question.

## Requirements


```bash
pip install -r requirements.txt
```

create a .env file and add the key as given below :

```text
GEMINI_API_KEY=YOUR_KEY
```


## Build the offline index

From the repository root:

```bash
python scripts/build_index.py
```

## Run the app

```bash
streamlit run app.py
```

The sidebar supports:

- Citizen / Operator / Auto
- Optional service selection
- Debug display of retrieved chunks


