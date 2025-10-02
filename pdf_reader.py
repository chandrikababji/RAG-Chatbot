import fitz  # PyMuPDF

def read_pdf(file_path):
    text = ""
    pdf = fitz.open(file_path)
    for page in pdf:
        text += page.get_text()
    return text

if __name__ == "__main__":
    print(read_pdf(r"c:\Users\babji\Downloads\Sai_Raam_Resume.pdf"))  # Replace with a PDF path
