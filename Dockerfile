FROM python:3.9-slim

WORKDIR /app

COPY pyproject.toml setup.cfg ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

EXPOSE 8000

ENTRYPOINT ["mlship"]
CMD ["deploy", "/model/model.pkl"]