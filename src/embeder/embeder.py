import json
from pathlib import Path
import ollama


def embed_jsonl_file(
    input_path: str | Path,
    output_path: str | Path,
    model: str = "embeddinggemma",
    embedding_field: str = "embeddinggemma_vec",
    batch_size: int = 64,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    # truncate/create output file first
    with output_path.open("w", encoding="utf-8") as out:
        pass

    records_batch = []
    texts_batch = []

    with input_path.open("r", encoding="utf-8") as f, \
         output_path.open("a", encoding="utf-8") as out:

        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            records_batch.append(rec)
            texts_batch.append((rec.get("text") or "").strip())

            if len(records_batch) >= batch_size:
                _embed_and_write_batch(
                    records_batch, texts_batch, out,
                    model=model, embedding_field=embedding_field
                )
                records_batch = []
                texts_batch = []

        # last partial batch
        if records_batch:
            _embed_and_write_batch(
                records_batch, texts_batch, out,
                model=model, embedding_field=embedding_field
            )


def _embed_and_write_batch(records, texts, out_file, model, embedding_field):
    if not texts:
        return

    resp = ollama.embed(model=model, input=texts)
    vectors = resp["embeddings"]
    assert len(records) == len(vectors)

    for rec, vec in zip(records, vectors):
        # pick whichever date field you want; here we use `publisheddate`
        date_value = rec.get("published_date") or rec.get("dateadded") or rec.get("day")
        num_ment = rec.get("max_num_mentions")

        minimal = {
            "date": date_value,
            embedding_field: vec,
            "num_mentions": num_ment
        }

        out_file.write(json.dumps(minimal, ensure_ascii=False) + "\n")