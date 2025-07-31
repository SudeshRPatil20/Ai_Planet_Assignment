# app/pdf_reader.py
from PyPDF2 import PdfReader

def parse_pdf_to_dict(pdf_path):
    reader = PdfReader(pdf_path)
    pages = reader.pages
    data = []
    qa_block = {}
    for page in pages:
        text = page.extract_text()
        blocks = text.split("\n")
        for line in blocks:
            if line.startswith("Question:"):
                if qa_block:
                    data.append(qa_block)
                    qa_block = {}
                qa_block["question"] = line.replace("Question:", "").strip()
            elif line.startswith("Solution:"):
                qa_block["solution"] = line.replace("Solution:", "").strip()
            elif line.startswith("Topic:"):
                qa_block["topic"] = line.replace("Topic:", "").strip()
            elif line.startswith("Difficulty:"):
                qa_block["difficulty"] = line.replace("Difficulty:", "").strip()
            elif line.startswith("Step"):
                qa_block.setdefault("steps", []).append(line.strip())
        if qa_block:
            data.append(qa_block)
    return data
