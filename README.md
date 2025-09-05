Python Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with:
	•	Google Gemini (for answering questions),
	•	Postgres + pgvector (for semantic search over document chunks),
	•	SentenceTransformers (all-MiniLM-L6-v2) for embeddings,
    •   pdfplumber + pdf2image + pytesseract – to extract text from PDFs (with OCR for scanned files)
	•	Streamlit (for the interactive UI).
Place the PDFs in your folder and then chat with them, the assistant will only use the PDFs in the folder as its knowledge source.


Features
	• Place the PDFs in the folder, and the app wil automatically extract their text 
    • Uses pdfplumber for text and tables
    • Uses Poppler + Tesseract to handle scanned PDFs
	• Store embeddings in Postgres with pgvector.
	• Perform semantic search on document chunks.
	• Query Gemini to answer questions grounded only in the PDF.
	• Refuses to answer questions outside the PDFs.
	• Simple Streamlit UI for chatting with your documents.


Installation
   • Install external tools (required for OCR):
	    	Poppler – (https://github.com/oschwartz10612/poppler-windows/releases/)
                Extract and add the Library\bin folder to your PATH.
	    	Tesseract OCR – (https://github.com/UB-Mannheim/tesseract/wiki/)
                Install and ensure tesseract.exe is on your PATH
   • Create & activate virtual environment:
        python -m venv venv
        .\venv\Scripts\Activate.ps1 (for windows) 
        pip install -r requirements.txt

   • Set up environment variables in a .env file:
        PGHOST=localhost
        PGPORT=5432
        PGDATABASE=your_db
        PGUSER=your_user
        PGPASSWORD=your_password
        GEMINI_API_KEY=your_api_key_here
        GEMINI_MODEL=gemini-2.5-flash
        DATA_FOLDER=your_folder_name

To Run the App
    streamlit run app.py


Possible errors
    pgvector not found!
        Install pgvector (https://github.com/pgvector/pgvector)
    No text extracted from PDF
        Your PDF might be scanned, ensure Tesseract is installed and on PATH.
    Gemini API error
        Make sure Gemini_API_Key is set in your .env file.