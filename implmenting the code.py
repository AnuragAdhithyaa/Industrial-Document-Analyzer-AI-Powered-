from google.colab import files

uploaded = files.upload()
for filename, filebytes in uploaded.items():
    print("\n--- Processing:", filename)
    if filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf_bytes(filebytes)
    elif filename.lower().endswith('.txt'):
        text = extract_text_from_txt_bytes(filebytes)
    else:
        print("Unsupported file type:", filename)
        continue

    print("Extracted characters:", len(text))
    print("\n--- Preview (first 800 chars) ---\n")
    print(text[:800])

    summary = summarize_long_text(text)
    print("\n SUMMARY \n")
    print(summary)

    keywords = extract_keywords(text, max_keywords=12)
    print("\n KEYWORDS \n")
    for kw, score in keywords:
        print(f"{kw} ({score:.4f})")
