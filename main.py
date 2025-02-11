from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import pdfplumber
# Load the model and tokenizer
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Create a QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

import sqlite3

# Connect to SQLite database (or create if it doesn't exist)
conn = sqlite3.connect("resumes.db")
cursor = conn.cursor()

# Create a table to store resume data
cursor.execute("""
CREATE TABLE IF NOT EXISTS resumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    phone TEXT,
    skills TEXT
)
""")
conn.commit()


def extract_text_from_pdf(pdf_path):


    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_info(question, context):
    result = qa_pipeline({"question": question, "context": context})
    return result["answer"]

import gradio as gr

def parse_resume(pdf_file):
    text = extract_text_from_pdf(pdf_file.name)
    return {
        "Name": extract_info("What is the candidate's name?", text),
        "Email": extract_info("What is the candidate's email?", text),
        "Phone": extract_info("What is the candidate's phone number?", text),
        "Skills": extract_info("What are the candidate's skills?", text),
    }

iface = gr.Interface(fn=parse_resume, inputs="file", outputs="json")
iface.launch()

