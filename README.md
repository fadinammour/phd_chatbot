# phd_chatbot
A chatbot to answer all your questions based on my PhD work!

Inspired and adapted from: https://python.langchain.com/v0.2/docs/how_to/

# Requirements

1. Download the PhD manuscript from the following link and place it in `./data` : https://theses.fr/2021UNIP7035
2. Install Ollama : https://ollama.com/download
3. Make sure that the LLM model in `./config.json` is pulled in Ollama. If the model is not pulled run `ollama run wizardlm2:7b`.
4. Install environment by running `poetry install` from root directory.

# Run chatbot
To launch the chatbot, run the following command from the root directory:
```python
poetry run python app.py
```
It will also provide a shareable temporary (72 hours) link that make the chatbot accessible on any device.