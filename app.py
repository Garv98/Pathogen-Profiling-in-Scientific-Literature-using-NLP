import streamlit as st
import pdfplumber
import torch
from transformers import pipeline, BartTokenizer
from PIL import Image
import io
import os
import re

@st.cache_resource
def load_summarizer():
    # Ensure the model uses the GPU if available, else fallback to CPU
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Extract text and structured content from the PDF
def extract_text_and_graphs(pdf_file):
    extracted_text = ""
    structured_content = {}
    figures = []
    current_section = None
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # Extract text
            text = page.extract_text()
            if text:
                extracted_text += text
                # Identify headings using regex
                lines = text.split("\n")
                for line in lines:
                    if re.match(r"^\d+\.\s+.*", line):  # Matches "1. Introduction" style headings
                        current_section = line.strip()
                        structured_content[current_section] = []
                    elif current_section:
                        structured_content[current_section].append(line.strip())
            # Extract images/graphs
            for image in page.images:
                figures.append(image)
    return extracted_text, structured_content, figures

# Save figures as image files
def save_figures(figures, pdf_file, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    figure_paths = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            for idx, image in enumerate(page.images):
                image_data = image.get("image")
                if image_data:
                    img = Image.open(io.BytesIO(image_data))
                    img_path = os.path.join(output_dir, f"figure_{page_num + 1}_{idx + 1}.png")
                    img.save(img_path)
                    figure_paths.append(img_path)
    return figure_paths

# Summarize sections efficiently
def summarize_sections(structured_content, summarizer):
    summaries = {}
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    # Ensure the summarizer model is on the same device as the input
    device = summarizer.device  # Get the device (either CPU or GPU)
    
    for section, content in structured_content.items():
        section_text = " ".join(content)

        # Check if text length exceeds the model's token limit (1024 tokens)
        if len(section_text.split()) > 1024:
            chunks = [section_text[i:i + 1024] for i in range(0, len(section_text), 1024)]
            full_summary = ""
            for chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=1024)
                # Move the inputs to the same device as the model
                inputs = {key: value.to(device) for key, value in inputs.items()}
                summary_ids = summarizer.model.generate(**inputs)
                summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                full_summary += summary_text + " "
            summaries[section] = full_summary.strip()
        else:
            inputs = tokenizer(section_text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
            # Move the inputs to the same device as the model
            inputs = {key: value.to(device) for key, value in inputs.items()}
            summary_ids = summarizer.model.generate(**inputs)
            summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries[section] = summary_text

    return summaries

# Highlight Extraction
def extract_highlights(structured_content):
    highlights = {}
    for section, content in structured_content.items():
        if "Dataset" in section or "Results" in section or "Conclusion" in section:
            highlights[section] = " ".join(content[:5])  # Extract top lines for highlights
    return highlights

# Main Streamlit App
def main():
    st.title("AI-Driven Research Paper Summarizer")
    st.write("Upload a scientific paper (PDF) to extract a detailed summary and visualize key insights.")

    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        st.subheader("Uploaded File:")
        st.write(uploaded_file.name)

        # Extract content
        with st.spinner("Extracting content from the PDF..."):
            try:
                extracted_text, structured_content, figures = extract_text_and_graphs(uploaded_file)
            except Exception as e:
                st.error(f"Error extracting content: {e}")
                return

        # Display extracted figures
        st.subheader("Extracted Figures and Graphs")
        if figures:
            st.write("Figures extracted from the paper:")
            figure_paths = save_figures(figures, uploaded_file)
            for fig_path in figure_paths:
                img = Image.open(fig_path)
                st.image(img, caption=os.path.basename(fig_path), use_column_width=True)
        else:
            st.write("No figures or graphs found in the paper.")

        # Summarize extracted text
        if extracted_text.strip():
            st.subheader("Summary of the Paper")
            with st.spinner("Summarizing the paper content..."):
                summarizer = load_summarizer()
                section_summaries = summarize_sections(structured_content, summarizer)
                for section, summary in section_summaries.items():
                    st.write(f"### {section}")
                    st.write(summary)

            # Display Highlights
            st.subheader("Key Highlights")
            highlights = extract_highlights(structured_content)
            for section, highlight in highlights.items():
                st.write(f"### {section}")
                st.write(f"- {highlight}")
        else:
            st.write("No text found for summarization.")

    # Sidebar information
    st.sidebar.title("About")
    st.sidebar.info("This tool uses advanced NLP and AI to summarize research papers and extract graphical content.")
    st.sidebar.markdown("[Learn more about BART](https://huggingface.co/facebook/bart-large-cnn)")

if __name__ == "__main__":
    main()
