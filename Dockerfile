# Containerizes the Streamlit app for portability/local-run demonstration --
# not how it's actually deployed (that's Streamlit Community Cloud, direct
# from GitHub, see sonic_explorer_spec.md section 10). Base deps only, same
# reasoning as requirements.txt: the deployed/containerized app only ever
# reads precomputed artifacts, it never runs CLAP/Demucs/librosa inference
# itself, so torch/transformers/demucs (the [colab] extra) are deliberately
# not installed here -- keeps the image small and the build fast.
FROM python:3.13-slim

WORKDIR /app

# System deps for faiss-cpu / scipy wheels that occasionally need a compiler
# fallback on slim images; kept minimal on purpose.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY sonic_explorer ./sonic_explorer
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app ./streamlit_app
COPY deploy_data ./deploy_data
COPY .streamlit ./.streamlit

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app/Overview.py", "--server.port=8501", "--server.address=0.0.0.0"]
