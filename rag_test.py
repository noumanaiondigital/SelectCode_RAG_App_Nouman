import os
import pandas as pd
import numpy as np
from trulens_eval import Feedback, TruChain
from trulens_eval.schema.feedback import Select
from trulens_eval.utils.display import get_feedback_result
from rag_utils import get_chunks, get_embeddings_and_retrievers
from trulens_eval import OpenAI as fOpenAI
import logging
from logging_config import setup_logging  # Import your logging configuration

# Set up logging
setup_logging()

logger = logging.getLogger(__name__)


def test_retrievers(ensemble_retriever, questions):
    """
    Test the performance of ensemble retrievers using a set of questions and record feedback results.

    Args:
        ensemble_retriever: The retriever object used for retrieving relevant contexts.
        questions (list): List of questions to test the retriever.

    Returns:
        None
    """
    try:
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
            try:
                if os.path.isfile(file_path):
                    existing_df = pd.read_csv(file_path)
                    updated_df = pd.concat([existing_df, get_feedback_result(all_records, "Context Relevance")],
                                           ignore_index=True)
                else:
                    updated_df = get_feedback_result(all_records, "Context Relevance")
                updated_df.to_csv(file_path, index=False)
            except Exception as e:
                logger.error(f"Error saving context relevance feedback results: {e}")
                raise

            # Save answer relevance feedback results
            file_path = "answer_relevance.csv"
            try:
                if os.path.isfile(file_path):
                    existing_df = pd.read_csv(file_path)
                    updated_df = pd.concat([existing_df, get_feedback_result(all_records, "Answer Relevance")],
                                           ignore_index=True)
                else:
                    updated_df = get_feedback_result(all_records, "Answer Relevance")
                updated_df.to_csv(file_path, index=False)
            except Exception as e:
                logger.error(f"Error saving answer relevance feedback results: {e}")
                raise

        logger.info("Retriever testing completed successfully.")
    except Exception as e:
        logger.error(f"Error testing retrievers: {e}")
        raise


def run_rag_test(file_path: str, questions_file_path: str) -> None:
    """
    Run a RAG (Retrieval-Augmented Generation) test on a given document and questions.

    Args:
        file_path (str): Path to the document file.
        questions_file_path (str): Path to the file containing evaluation questions.

    Returns:
        None
    """
    try:
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

        test_retrievers(ensemble_retriever, eval_questions)

        logger.info("RAG test run completed successfully.")
    except Exception as e:
        logger.error(f"Error running RAG test: {e}")
        raise
