# Video-Query-Agent

A simple pipeline to build and query video data using Qdrant and AI models.

---

## Getting Started

### 1. Run Qdrant with Docker
Make sure you have [Docker](https://docs.docker.com/get-docker/) installed.  
Start Qdrant with `docker-compose`:

```bash
docker-compose up -d
```

- This will spin up a Qdrant instance on ports `6333` (REST API) and `6334` (gRPC API).  
- By default, data will be mounted to `./qdrant_storage`.  
- If you **don’t want to persist data**, just comment out the `volumes` section in `docker-compose.yml`.

Check if Qdrant is running:

```bash
docker ps
```

---

### 2. Prepare Raw Data
Copy all your data folders into:

```
raw_data/
```

---

### 3. Set Up API Key
1. Go to [Google AI Studio](https://ai.google.dev/) and generate an API key.  
2. Copy the key and paste it into `.env.example`.  
3. Rename the file to `.env`.

---

### 4. Install Dependencies
Make sure you have **Python 3.9+** installed.  
Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

### 5. Build Databases
Run the builder script to process raw data and populate Qdrant:

```bash
python builder/run_builder.py
```

---

### 6. Run the Application
Finally, start the main application:

```bash
python main.py
```

---

## Summary
1. Start Qdrant (`docker-compose up -d`)  
2. Place your data in `raw_data/`  
3. Add your API key to `.env`  
4. Install dependencies  
5. Build databases with `run_builder.py`  
6. Launch with `main.py`  

---

## Notes
- Make sure Docker is running before executing `docker-compose up`.  
- If you stop and restart your computer, Qdrant will automatically restart (thanks to the `restart: always` option in `docker-compose.yml`).  
- To stop the container, run:

```bash
docker-compose down
```
