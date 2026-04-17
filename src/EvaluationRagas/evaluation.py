import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from retriever import RetrieveChunks
from EmbeddingModel import EmbModel
from helpers import config
from dotenv import load_dotenv

from llm.llm import get_llm
load_dotenv()
settings = config.get_settings()

llm = get_llm()

# def run_rag_evaluation(vector_store):
    
#     test_questions = [
#         "What are the Major ?",
#         "List the internships .",
#         "Which bank is mentionted ?"
#     ]

#     samples = []
   
#     embeddings = EmbModel.get_embedding()


#     for question in test_questions:
#         retrieved_docs = RetrieveChunks.advanced_retrieve(query=question)
#         contexts = [doc.page_content for doc in retrieved_docs]
        

#         context_text = "\n\n".join(contexts)
#         full_prompt = f"Context: {context_text}\n\nQuestion: {question}\n\nAnswer:"
#         response = llm.invoke(full_prompt)

#         samples.append({
#             "user_input": question,
#             "retrieved_contexts": contexts,
#             "response": response.content,
#         })

#     dataset = EvaluationDataset.from_list(samples)

#     ragas_llm = LangchainLLMWrapper(llm)
#     ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

#     results = evaluate(
#         dataset=dataset,
#         metrics=[Faithfulness(), ResponseRelevancy()],
#         llm=ragas_llm,
#         embeddings=ragas_embeddings,
#     )

#     return results.to_pandas()
def run_rag_evaluation():
    test_questions = [
        "What are the Major ?",
        "List the internships .",
        "Which bank is mentioned ?"
    ]

    samples = []
    embeddings = EmbModel.get_embedding()

    for question in test_questions:
        # استرجاع السياق
        retrieved_docs = RetrieveChunks.advanced_retrieve(query=question)
        contexts = [doc.page_content for doc in retrieved_docs]
        
        # توليد الإجابة
        context_text = "\n\n".join(contexts)
        full_prompt = f"Context: {context_text}\n\nQuestion: {question}\n\nAnswer:"
        response = llm.invoke(full_prompt)

        samples.append({
            "user_input": question,
            "retrieved_contexts": contexts, # Ragas يتوقع قائمة نصوص
            "response": response.content,
        })

    # تحويل البيانات إلى Dataset الخاص بـ Ragas
    dataset = EvaluationDataset.from_list(samples)

    # تغليف الموديلات لتتوافق مع Ragas
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # إجراء التقييم
    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    return results.to_pandas()
