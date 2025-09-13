# Repository Setup and Usage

## Prerequisites

- Python **3.12**
- Docker
- **Ripgrep** (for text search functionality)
- (Optional) GPU for faster vector DB building

---

## 1. Environment Setup

### Install Ripgrep

**Windows:**

```bash
# Option 1: Using pip (recommended)
pip install ripgrep

# Option 2: Using Chocolatey (requires admin)
choco install ripgrep

# Option 3: Using Scoop
scoop install ripgrep
```

**Linux/macOS:**

```bash
# Ubuntu/Debian
sudo apt install ripgrep

# macOS
brew install ripgrep
```

### Python Setup

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

### DevOps Notice:

Do not run this script if the configuration is set to use a cloud database. Run it only when you explicitly intend to build new embeddings and are certain of the implications.

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

---

## PromptLayer Integration

- Add `PROMPT_LAYER_API_KEY` to a `.env` file in the repo root.
- Default PromptLayer prompt name is `Neti_repo_assistant`.
- Optional override: set `PROMPTLAYER_PROMPT_NAME` in `.env`.
- When `PROMPT_LAYER_API_KEY` is present, the app uses PromptLayer for the system prompt and routes the call through PromptLayer with tool-calling enabled. Otherwise it falls back to the YAML prompt in `configs/prompt_config.yaml`.
