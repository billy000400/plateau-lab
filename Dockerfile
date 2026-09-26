# Image for a Hugging Face Docker Space (or any container host): API + static frontend.
FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
# CPU wheel: inference runs on NDIF, so the server needs no GPU build.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY --chown=user plateau plateau
COPY --chown=user server server
COPY --chown=user web web
# Runs on NDIF with each user's own key (no server key needed).
ENV PLATEAU_REMOTE=1
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
