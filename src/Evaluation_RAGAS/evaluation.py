import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from retriever import retrieve_chunks
from embedding_model import emb_model
from helpers import config
from dotenv import load_dotenv
from langchain_groq import ChatGroq



load_dotenv()
settings = config.get_settings()

llm = ChatGroq(
    model=settings.MODEL_NAME,
    api_key=settings.GROQ_API_KEY
)

def run_rag_evaluation(vector_store):
    test_questions = [
        "What are the Major ?",
        "List the internships .",
        "Which bank is mentionted ?"
    ]

    samples = []
   
    embeddings = emb_model.get_embedding()


    for question in test_questions:
        retrieved_docs = retrieve_chunks.advanced_retrieve(vector_store=vector_store, query=question)
        contexts = [doc.page_content for doc in retrieved_docs]
        

        context_text = "\n\n".join(contexts)
        full_prompt = f"Context: {context_text}\n\nQuestion: {question}\n\nAnswer:"
        response = llm.invoke(full_prompt)

        samples.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": response.content,
        })

    dataset = EvaluationDataset.from_list(samples)

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    return results.to_pandas()