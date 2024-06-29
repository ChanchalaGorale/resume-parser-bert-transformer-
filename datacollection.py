import os
import textract

def extract_text(file):
    return textract.process(file).decode("utf-8")


file="/resume.pdf"




text = extract_text(file)

resume = [text]


