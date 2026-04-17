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

load_dotenv()
settings = config.get_settings()


llm = get_llm()


def run_rag_evaluation():

    test_questions = [
        "What is the major of seif mahmoud ?",
        "what is email yassein ahmed ? .",
        "what is major youssab kamal  ?",
        "what is phone number of youssab kamel ?",
        "what is salma loves ?",
        "what is email of youssab kamal ?"
    ][:3]   

    embeddings = EmbModel.get_embedding()

    samples = []

    for question in test_questions:


        retrieved_docs = RetrieveChunks.advanced_retrieve(
            query=question
        )[:3]   

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