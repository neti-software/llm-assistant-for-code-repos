# Repository Setup and Usage

## Prerequisites
- Python **3.12**
- Docker
- (Optional) GPU for faster vector DB building

---

## 1. Environment Setup
```bash
# Create a virtual environment with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Start Vector Database (Qdrant via Docker)
```bash
docker run -d   --name qdrant   -p 7000:6333   --ulimit nofile=65536:65536   qdrant/qdrant
```

---

## 3. Download Repositories
```bash
python src/download_org_repos.py filecoin-project
python src/download_org_repos.py fiddler-labs
```

---

## 4. Configure Keys and Paths
- **OpenAI key**  
  Copy `openai_key_EXAMPLE.yaml` → `openai_key.yaml` and set your API key.  

- **Repos config**  
  Update `repos_config.yaml` with:  
  ```yaml
  path_to_repos: path/to/repos
  ```

---

## 5. Build the Vector Database
Estimated time: **10–30 minutes** (GPU recommended).  
```bash
python main_build_db.py
```

---

## 6. Run the Application

### CLI Mode
```bash
python main_cli.py
```

### Web Mode (Streamlit)
```bash
streamlit run main_streamlit.py
```