Data Directory Overview

This directory contains various data files utilized in the Self-RAG framework. Each file serves a specific purpose and has unique input requirements. Below is a brief description of each file:
	•	context_embeddings.pkl: This file contains embeddings for the FinQA dataset. These embeddings are generated using the generate_passage_embedding.py script.
	•	finqa_passages.tsv: A tab-separated values (TSV) file that holds passage data related to FinQA. Each row typically represents a passage used in the dataset.
	•	finqa_retriever_data.jsonl: A JSON Lines file containing data formatted for the retriever component. Each line is a JSON object representing an individual data point used during the retrieval process.
	•	output.jsonl: This file stores the output from the retriever. Each line is a JSON object containing the retrieval results for a specific input query.
	•	retriever_results.jsonl: A JSON Lines file that includes passages retrieved in context for the inference phase. Each line corresponds to a query and its associated retrieved passages and answers facilitating the inference process.

