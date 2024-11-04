from dotenv import load_dotenv
import warnings
import os

load_dotenv()

import openai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_openai import OpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Suppress all warnings
# Deprecation warning is annoying as hell, tried to fix but no luck.
warnings.filterwarnings("ignore")

model_name = "Helsinki-NLP/opus-mt-id-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=500,
    device=0
)

# OpenAI
openai.api_key = os.getenv("OPEN_API_KEY")

prompt_template = """
Your name is Lisa, you are a professional Indonesian-English teacher, and your task is to correct & summarize what your student conversation in
Indonesian conversation practice. If students make grammar or vocabulary mistakes, you have to correct student and provide the correct usage.
Then after you correct all the Indonesian grammar mistakes, you have to summarize what are the conversations are about.

Student conversations:
{inputs}
"""
open_prompt = PromptTemplate(input_variables=["inputs"], template=prompt_template)

llm = HuggingFacePipeline(pipeline=pipe)
t_prompt = PromptTemplate(input_variables=["text"], template="{text}")
t_chain = LLMChain(llm=llm, prompt=t_prompt)

history = []
print("Masukkan text atau ketik 'done' untuk selesai dan merangkum \n")
print("------------------- Percakapan Indo ke English ------------------")

while(True):
    txt = input("- ")

    if txt is None:
        continue
    elif txt.lower() == "done":
        print("Selesai... Ayo rangkum chat kita dalam dua bahasa... \n")
        break

    # not using memory because i just want the raw text from user as history
    history.append(txt)

    en_txt = t_chain.run({"text": txt})
    print("> ", en_txt, "\n")


open_chain = LLMChain(
    llm=OpenAI(api_key=openai.api_key),
    prompt=open_prompt,
)
summary = open_chain.run(inputs=history)

print("--------------------- RANGKUMAN ----------------------")
print(summary)