def get_md_content(documents):
    raw_text = ""

    for document in documents:
        content = document.read().decode('utf-8')
        raw_text += content + "\n"

    return raw_text