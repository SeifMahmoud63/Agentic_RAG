from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from retriever import RetrieveChunks
from EmbeddingModel import EmbModel
from helpers import config
from dotenv import load_dotenv
from llm.llm import get_llm
import time
import os
from logs.logger import logger

load_dotenv()
settings = config.get_settings()


llm = get_llm()


import asyncio

def run_rag_evaluation():

    # Load questions from external file
    test_questions = []
    test_data_path = os.path.join(os.path.dirname(__file__), "test_data.txt")
    if os.path.exists(test_data_path):
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_questions = [line.strip() for line in f if line.strip()]
    
    if not test_questions:
        logger.error(f"test_data.txt not found or empty at {test_data_path}. Please add questions to run evaluation.")
        import pandas as pd
        return pd.DataFrame()
    
    # We take the first 3 for efficiency in this run
    test_questions = test_questions[:3]

    embeddings = EmbModel.get_embedding()

    samples = []

    for question in test_questions:


        retrieved_docs = asyncio.run(RetrieveChunks.advanced_retrieve(
            query=question
        ))[:3]   

        MAX_CONTEXT_CHARS = 300

        contexts = [
            doc.page_content[:MAX_CONTEXT_CHARS]
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

        answer_text = answer.content[:600]

        samples.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer_text,
        })

        time.sleep(2)

    dataset = EvaluationDataset.from_list(samples)

    eval_llm = ChatGroq(
        model="llama-3.1-8b-instant",  
        temperature=0,
        max_tokens=1024,
        n=1
    )

    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(strictness=1)
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(
            max_workers=1,
            timeout=300,
            max_retries=20,
            max_wait=5 
        )
    )

    return results.to_pandas()