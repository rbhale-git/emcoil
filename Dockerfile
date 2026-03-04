FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user emcoil/ emcoil/
COPY --chown=user app.py .
COPY --chown=user assets/ assets/

EXPOSE 7860

CMD ["python", "app.py"]
