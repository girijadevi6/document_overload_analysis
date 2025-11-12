# document_overload_analysis
Objective:
To automate the extraction, processing, summarization, and classification of documents from multiple sources (emails, PDFs, Word files, images) to support faster decision-making and reduce manual workload.

What We Did:

Data Ingestion:

Integrated with Gmail to fetch unread emails and attachments.

Supported manual file uploads (PDF, DOCX, TXT, images).

Text Extraction:

Extracted text from PDFs using PyMuPDF, falling back to OCR (Tesseract) for scanned documents.

Extracted content from DOCX, TXT, and images with preprocessing for better accuracy.

Text Cleaning & Translation:

Cleaned text by removing unnecessary headers, footers, special characters.

Detected Malayalam text and translated it to English automatically using Google Translator.

Summarization:

Applied a hybrid summarization approach combining:

Extractive summarization (TextRank via Sumy) to pick important sentences.

Abstractive summarization (DistilBART) to generate concise summaries.

Classification & Urgency Detection:

Used zero-shot classification (DistilBERT) to categorize documents into domains (Finance, HR, Incident Reports, etc.).

Applied rules and ML to determine urgency, flagging critical documents for immediate attention.

Storage & Retrieval:

Stored all documents, metadata, summaries, and classifications in SQLite database.

Built a Flask web dashboard for users to view, filter, and download documents.

Alerting System:

Automatically sent email alerts to domain heads for urgent documents.

Technologies Used:

Backend: Python, Flask, SQLite

NLP & ML: Transformers (DistilBART, DistilBERT), spaCy, sumy, langdetect, googletrans

OCR & Image Processing: Tesseract, OpenCV, PyMuPDF, PIL

Frontend: Flask templates (HTML/CSS), dashboard for document viewing

Email Integration: IMAP/SMTP for fetching emails and sending alerts

Outcome:

Fully automated pipeline for document processing from ingestion to summarization, classification, and alerting.

Reduced manual effort, improved information retrieval, and ensured critical documents are prioritized.
