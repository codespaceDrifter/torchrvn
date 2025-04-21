from datasets import load_from_disk

def dataset_to_txt(dataset_path, output_txt_path, text_key="text"):
    ds = load_from_disk(dataset_path)

    with open(output_txt_path, "w", encoding="utf-8") as out_file:
        for i, example in enumerate(ds):
            text = example.get(text_key, "").strip()
            if text:
                out_file.write(text + "\n\n")
            if i % 1000 == 0:
                print(f"Wrote {i} examples...")

    print(f"✅ Done. Wrote all text to {output_txt_path}")