
import os
import pandas as pd
import numpy as np
from trulens_eval import Feedback, TruChain
from trulens_eval.schema.feedback import Select
from trulens_eval.utils.display import get_feedback_result
from rag_utils import get_chunks, get_embeddings_and_retrievers
from trulens_eval import OpenAI as fOpenAI



def test_retrievers(ensemble_retriever, questions):
    ensemble_context = Select.RecordCalls.invoke.rets[:].page_content
    provider = fOpenAI(model_engine='gpt-4o-mini')

    f_groundedness = (
        Feedback(provider.groundedness_measure_with_cot_reasons,
                name="Groundedness"
                )
        .on(ensemble_context)
        .on_output()
        .aggregate(np.mean)
    )
    #Answer relevance
    f_qa_relevance = Feedback(
        provider.relevance_with_cot_reasons,
        name="Answer Relevance"
    ).on_input_output()


    #Context Relevance
    f_qs_relevance = (
        Feedback(provider.qs_relevance_with_cot_reasons,
                name="Context Relevance")
        .on_input()
        .on(ensemble_context)
        .aggregate(np.mean)
    )


    tru_recorder = TruChain(ensemble_retriever,
        app_id='Ensemble Retriever',
        feedbacks=[f_qa_relevance,
                   f_qs_relevance
    ])

    for question_query in questions:

        with tru_recorder as recording:
            ensemble_retriever.invoke(question_query)
        all_records = recording.records[-1]

        file_path = "context_relevance.csv"

        # Check if the file exists
        if os.path.isfile(file_path):
            # Load the existing CSV file
            existing_df = pd.read_csv(file_path)

            # Append the new data to the existing DataFrame
            updated_df = pd.concat([existing_df, get_feedback_result(all_records, "Context Relevance")], ignore_index=True)
        else:
            # If the file does not exist, the updated DataFrame is the new data
            updated_df = get_feedback_result(all_records, "Context Relevance")
        updated_df.to_csv(file_path, index=False)

        file_path = "answer_relevance.csv"

        # Check if the file exists
        if os.path.isfile(file_path):
            # Load the existing CSV file
            existing_df = pd.read_csv(file_path)

            # Append the new data to the existing DataFrame
            updated_df = pd.concat([existing_df, get_feedback_result(all_records, "Answer Relevance")], ignore_index=True)
        else:
            # If the file does not exist, the updated DataFrame is the new data
            updated_df = get_feedback_result(all_records, "Answer Relevance")
        updated_df.to_csv(file_path, index=False)

    
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


def run_rag_test(file_path, questions_file_path):

    # Read the document
    with open(file_path, 'r', encoding='utf-8') as document:
        test_doc_content = document.read()
    
    eval_questions = []
    with open(questions_file_path, 'r') as file:
        for line in file:
            # Remove newline character and convert to integer
            item = line.strip()
            eval_questions.append(item)

    # convert text to chunks of data
    text_chunks = get_chunks(test_doc_content)

    ensemble_retriever = get_embeddings_and_retrievers(text_chunks)

    from rag_test import test_retrievers
    test_retrievers(ensemble_retriever, eval_questions)
