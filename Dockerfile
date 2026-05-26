FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/main.py .
COPY backend/inference.py .
COPY backend/schemas.py .
COPY backend/utils.py .
COPY backend/models/ ./models/

# Copy built frontend static files
COPY frontend/out/ ./static/

# Patch main.py to also serve static frontend
RUN python3 - << 'PYEOF'
content = open("main.py").read()
addition = """
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Serve Next.js static export
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static_assets")

@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse("static/index.html")

@app.get("/{path:path}", include_in_schema=False)
async def serve_spa(path: str):
    file_path = f"static/{path}"
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    index = f"static/{path}/index.html"
    if os.path.exists(index):
        return FileResponse(index)
    return FileResponse("static/index.html")
"""
# Insert before last line or append
open("main.py", "a").write(addition)
PYEOF

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
