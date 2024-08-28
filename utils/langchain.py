import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from accelerate import Accelerator
import torch


def load_hf_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def load_hf_auto_regressive_model(model_name: str, load_in_4bit=False, load_in_8bit=False,
                                  max_new_tokens=512, cache_dir=None) -> HuggingFacePipeline:

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
        )
        device_map = 'auto'
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
    )

    pip = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pip)


def load_llamacpp_embedding_model(base_dir='./models/', model_file='nomic-embed-text-v1.5.f16.gguf'):
    return LlamaCppEmbeddings(model_path=os.path.join(base_dir, model_file), n_batch=512)


def load_ollama_autoregressive_model(model_name, json=False):
    if json:
        return ChatOllama(model=model_name, format='json')
    else:
        return ChatOllama(model=model_name)