import os
import pandas as pd
import numpy as np
from trulens_eval import Feedback, TruChain
from trulens_eval.schema.feedback import Select
from trulens_eval.utils.display import get_feedback_result
from rag_utils import get_chunks, get_embeddings_and_retrievers
from trulens_eval import OpenAI as fOpenAI

def test_retrievers(ensemble_retriever, questions):
    """
    Test the performance of ensemble retrievers using a set of questions and record feedback results.

    Args:
        ensemble_retriever: The retriever object used for retrieving relevant contexts.
        questions (list): List of questions to test the retriever.

    Returns:
        None
    """
    ensemble_context = Select.RecordCalls.invoke.rets[:].page_content
    provider = fOpenAI(model_engine='gpt-4o-mini')
    
    # Answer relevance feedback
    f_qa_relevance = Feedback(
        provider.relevance_with_cot_reasons,
        name="Answer Relevance"
    ).on_input_output()

    # Context relevance feedback
    f_qs_relevance = (
        Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
        .on_input()
        .on(ensemble_context)
        .aggregate(np.mean)
    )

    tru_recorder = TruChain(
        ensemble_retriever,
        app_id='Ensemble Retriever',
        feedbacks=[f_qa_relevance, f_qs_relevance]
    )

    for question_query in questions:
        with tru_recorder as recording:
            ensemble_retriever.invoke(question_query)
        all_records = recording.records[-1]

        # Save context relevance feedback results
        file_path = "context_relevance.csv"
        if os.path.isfile(file_path):
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, get_feedback_result(all_records, "Context Relevance")], ignore_index=True)
        else:
            updated_df = get_feedback_result(all_records, "Context Relevance")
        updated_df.to_csv(file_path, index=False)

        # Save answer relevance feedback results
        file_path = "answer_relevance.csv"
        if os.path.isfile(file_path):
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, get_feedback_result(all_records, "Answer Relevance")], ignore_index=True)
        else:
            updated_df = get_feedback_result(all_records, "Answer Relevance")
        updated_df.to_csv(file_path, index=False)

    # Uncomment the following lines for additional testing and logging as needed

    # from trulens_eval import Tru
    # tru = Tru()
    # tru.reset_database()
    # records, feedback = tru.get_records_and_feedback(app_ids=[])
    # last_record = recording.records[-1]

    # print("result of Context")
    # print(type(get_feedback_result(last_record, "Context Relevance")))
    # print(get_feedback_result(last_record, "Context Relevance"))
    # print("result of Answer Relevance")
    # print(get_feedback_result(last_record, "Answer Relevance"))
    # print(get_feedback_result(last_record, "Groundedness"))

    # print(records.head())
    # records.to_csv("test_results.csv", index=False)
    # records.head().to_csv("test_results1.csv", index=False)

    # pd.set_option("display.max_colwidth", None)
    # records[["input", "output"] + feedback]

    # tru.get_leaderboard(app_ids=[])
    # tru.run_dashboard()

def run_rag_test(file_path: str, questions_file_path: str) -> None:
    """
    Run a RAG (Retrieval-Augmented Generation) test on a given document and questions.

    Args:
        file_path (str): Path to the document file.
        questions_file_path (str): Path to the file containing evaluation questions.

    Returns:
        None
    """
    # Read the document
    with open(file_path, 'r', encoding='utf-8') as document:
        test_doc_content = document.read()
    
    # Read evaluation questions from the file
    eval_questions = []
    with open(questions_file_path, 'r') as file:
        for line in file:
            item = line.strip()
            eval_questions.append(item)

    # Convert text to chunks of data
    text_chunks = get_chunks(test_doc_content)

    # Get embeddings and create ensemble retriever
    ensemble_retriever = get_embeddings_and_retrievers(text_chunks)

    from rag_test import test_retrievers
    test_retrievers(ensemble_retriever, eval_questions)
