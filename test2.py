import os


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

PATH="/home/gridsan/pnegi/hf/models--philschmid--flan-t5-xxl-sharded-fp16"
model = AutoModelForSequenceClassification.from_pretrained(PATH)

