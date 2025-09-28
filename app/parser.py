import fitz  # PyMuPDF

def extract_text_from_pdf(file):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in pdf)
