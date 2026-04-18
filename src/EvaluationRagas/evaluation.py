from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from retriever import retrieve_chunks as RetrieveChunks
from embeddingmodel import emb_model as EmbModel
from helpers import config
from dotenv import load_dotenv
from llm.llm import get_llm
import time
import pandas as pd
import os
from logs.logger import logger

load_dotenv()
settings = config.get_settings()


llm = get_llm()


import asyncio

def run_rag_evaluation():

    test_questions = []
    test_data_path = os.path.join(os.path.dirname(__file__), "test_data.txt")
    if os.path.exists(test_data_path):
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_questions = [line.strip() for line in f if line.strip()]
    
    if not test_questions:
        logger.error(f"test_data.txt not found or empty at {test_data_path}. Please add questions to run evaluation.")
        
        return pd.DataFrame()
    
    test_questions = test_questions[:settings.EVAL_MAX_QUESTIONS]

    embeddings = EmbModel.get_embedding()

    samples = []

    for question in test_questions:

        retrieved_docs = asyncio.run(RetrieveChunks.advanced_retrieve(
            query=question
        ))[:settings.EVAL_RETRIEVAL_K]   

        contexts = [
            doc.page_content[:settings.EVAL_MAX_CONTEXT_CHARS]
            for doc in retrieved_docs
        ]

        context_text = "\n\n".join(contexts)

        full_prompt = f"""
Context:
{context_text}

Question:
{question}

Answer:
"""

        answer = llm.invoke(full_prompt)

        answer_text = answer.content[:settings.EVAL_MAX_RESPONSE_CHARS]

        samples.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer_text,
        })

        time.sleep(settings.EVAL_SLEEP_BETWEEN_SAMPLES)

    dataset = EvaluationDataset.from_list(samples)

    eval_llm = ChatGroq(
        model=settings.EVAL_LLM_MODEL,  
        temperature=settings.EVAL_LLM_TEMPERATURE,
        max_tokens=settings.EVAL_LLM_MAX_TOKENS,
        n=1
    )

    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(strictness=settings.EVAL_RELEVANCY_STRICTNESS)
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(
            max_workers=settings.EVAL_MAX_WORKERS,
            timeout=settings.EVAL_TIMEOUT,
            max_retries=settings.EVAL_MAX_RETRIES,
            max_wait=settings.EVAL_MAX_WAIT 
        )
    )

    return results.to_pandas()