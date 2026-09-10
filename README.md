# FairGraphRAG

FairGraphRAG is a fairness-aware Graph Retrieval-Augmented Generation (GraphRAG) prototype for answering questions about occupations. It combines structured O*NET career data, a multi-relational knowledge graph, semantic vector retrieval, fairness-aware reranking, and a language model that writes answers from retrieved evidence.

The project is designed to study whether occupational answers remain useful and consistent when the wording of a question changes the person described by it.

## Project Goals

- Retrieve occupational information from a large structured career dataset.
- Use relationships between occupations and their requirements to improve retrieval.
- Give a language model relevant evidence instead of asking it to answer from memory.
- Check whether retrieval and generated answers change unfairly for comparable questions.
- Compare a basic vector approach, GraphRAG, and fairness-aware GraphRAG.

## Data Used

The notebook uses the O*NET 30.0 database available in the Kaggle environment. The following tables are loaded:

- `occupation_data.csv`: occupation codes, titles, and descriptions.
- `essential_skills.csv`: skills and their importance for occupations.
- `knowledge.csv`: knowledge areas and their importance.
- `education.csv`: education requirements and related values.
- `software_skills.csv`: software and technology requirements.
- `task_statements.csv`: tasks performed in an occupation.
- `related_occupations.csv`: relationships between occupations.

Missing values are replaced with empty strings. Skill and knowledge records are filtered to the importance scale (`Scale ID == "IM"`) and records marked for suppression are excluded. Lookup dictionaries connect O*NET-SOC codes, titles, and descriptions.

## Implemented Pipeline

### 1. Data preparation

The notebook reads the O*NET CSV files with pandas, checks their columns and sizes, removes missing values, filters the important skill and knowledge records, and creates lookup structures for later graph construction and retrieval.

### 2. Knowledge graph construction

The graph is built with `networkx.MultiDiGraph`, which supports directed edges and multiple relationship types between the same nodes.

The graph includes these node types:

- Occupations
- Skills
- Knowledge areas
- Education levels
- Software and technology
- Tasks

The graph connects occupations to their skills, knowledge, education, software, and tasks. It also adds `RELATED_TO` edges between related occupations. Edge attributes preserve useful information such as importance values, weights, relation names, and task metadata.

The notebook calculates graph statistics and PageRank values. PageRank is stored as a node attribute and is later used as one signal during graph-based ranking. The completed graph is saved as `fairgraphrag_onet_graph.pkl` in the Kaggle working directory.

### 3. Occupation text corpus

For every occupation, the notebook creates a searchable text record containing the occupation title, description, important skills, knowledge areas, education information, software, and tasks. This gives the semantic retriever both the occupation summary and its connected requirements.

### 4. Semantic embeddings and vector search

The embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The occupation text records are converted into normalized embeddings. FAISS is used with `IndexFlatIP`, so inner product on normalized vectors acts as cosine similarity. The index returns the top matching occupations for a user query.

The following artifacts are saved for reuse:

- `occupation_embeddings.npy`
- `occupation_faiss.index`

The main vector retrieval function is `vector_retrieve(query, top_k=5)`. It returns the matching occupation title, O*NET code, graph node, and similarity score.

### 5. Query-aware graph expansion

GraphRAG starts with the top occupations returned by FAISS. The `graph_expand` function then explores their neighboring nodes and gathers connected occupations through the graph.

Each candidate receives a combined graph score based on:

- The original semantic similarity of the seed occupation.
- The importance of the relationship connecting the candidate.
- The candidate node's PageRank.
- The number and strength of graph connections.

The notebook retains the original vector candidates, adds graph neighbors, ranks candidates, and limits the expanded graph to a configurable number of nodes. The default GraphRAG flow uses up to 80 graph nodes for the final context.

### 6. GraphRAG context building

The `graphrag_retrieve` function combines vector retrieval and graph expansion. It returns:

- Initial vector seed results.
- Expanded graph nodes.
- Candidate scores.
- Ranked occupations.
- A final text context for the language model.

The `retrieve_with_metadata` function adds inspection details such as graph nodes visited, graph edges traversed, top similarity, and average similarity. These values make it possible to analyze not only the answer but also how the answer's evidence was retrieved.

## Fairness and Bias Evaluation

### Counterfactual query pairs

The notebook creates matched question pairs for sampled occupations. Each pair asks the same occupational question with a different protected or demographic description:

- Gender: man and woman
- Religion: Christian and Muslim
- Age: young and old person
- Nationality: American and Indian
- Disability: non-disabled and disabled person

The purpose is to test whether changing only this description causes an unnecessary change in retrieved occupations, rankings, graph coverage, or generated decisions.

### Retrieval fairness metrics

The notebook implements functions for:

- `representation_balance`: compares the retrieved occupation sets for a query pair.
- `ranking_fairness`: measures average rank differences between paired retrievals.
- `counterfactual_consistency`: checks whether comparable queries produce consistent top results.
- `retrieval_bias_score`: combines representation and counterfactual signals.
- `exposure_fairness`: measures how often occupations are exposed across retrieved lists.

It also records similarity gaps, graph node coverage gaps, graph edge coverage gaps, top-result similarity, and average similarity. Results are summarized overall and by bias type.

### Fairness-aware reranking

The fairness-aware version creates two retrieval lists for a counterfactual pair, measures occupation exposure and overlap, and adds fairness features to the candidates. The `fair_rerank` function adjusts ranking using:

- The original retrieval score.
- An overlap bonus for occupations appearing in both paired results.
- An exposure penalty to reduce overexposed items.
- A fairness adjustment that favors more consistent evidence across comparable queries.

The final `fairgraphrag_retrieve` and `build_fair_context` functions use this reranked evidence to build the context supplied to the language model.

## Language Model Answering

The notebook uses:

```text
Qwen/Qwen2.5-3B-Instruct
```

The tokenizer and causal language model are loaded with Hugging Face Transformers. A prompt is built from the user question and the retrieved occupation context. Generation is configured deterministically with `do_sample=False`, `temperature=0.0`, and `max_new_tokens=60` for the main GraphRAG answer workflow.

The notebook includes separate answer functions for the baseline GraphRAG path and the fairness-aware path. It also extracts a simple decision from each answer so paired answers can be compared.

## System Versions Compared

### M0: Vector baseline

M0 retrieves occupations directly from the FAISS semantic index without graph expansion. It provides the baseline for retrieval and language-model comparisons.

### M1: GraphRAG

M1 starts with vector retrieval, expands through the O*NET graph, ranks graph candidates, and sends the resulting context to the language model. It measures whether graph structure improves evidence coverage and answer quality.

### M2: FairGraphRAG

M2 adds counterfactual retrieval, exposure and overlap features, fairness-aware reranking, and fairness-aware context construction before answer generation. It is the main proposed approach in this notebook.

## Language Model Evaluation

The generated answers are evaluated using:

- Decision consistency across matched counterfactual questions.
- Faithfulness based on overlap between answer words and retrieved context words.
- Hallucination rate, calculated as `1 - faithfulness`.
- Context faithfulness.
- Protected-attribute mention rate.
- Average toxicity score.
- Answer length and extracted decision distributions.

The notebook runs sample and larger evaluation sets, saves checkpoints during long runs, and writes result tables to CSV files in the Kaggle working directory.

## Outputs and Saved Results

The notebook saves graph, embedding, index, retrieval, fairness, and answer artifacts, including:

- Pickled knowledge graph.
- NumPy occupation embeddings.
- FAISS occupation index.
- M1 GraphRAG retrieval metrics and bias summaries.
- M2 FairGraphRAG retrieval metrics and exposure summaries.
- Qwen GraphRAG answer results.
- FairGraphRAG answer results with context and evaluation columns.
- Checkpoint CSV files for long-running answer generation.

## Main Dependencies

The first notebook cell installs the main packages used by the project:

- `pandas` and `numpy` for data preparation and numerical processing.
- `networkx` for the multi-relational graph.
- `sentence-transformers` for occupation embeddings.
- `faiss-cpu` for vector similarity search.
- `transformers` and `torch` for language-model inference.
- `langchain`, `langchain-community`, and `langchain-text-splitters` for document utilities.
- `pypdf` for PDF loading utilities.
- `spaCy` for natural-language processing support.
- `scikit-learn` for cosine similarity calculations.
- `tqdm` for progress reporting.

## How to Run

1. Open [`ownrag-projectdone.ipynb`](ownrag-projectdone.ipynb).
2. Use a Kaggle notebook environment or download the same O*NET dataset locally.
3. If running outside Kaggle, update the `BASE_PATH` value near the beginning of the notebook.
4. Run the cells in order because later cells use the graph, embeddings, indexes, and data frames created earlier.
5. Expect the language-model and large counterfactual evaluation sections to take significantly longer than the data preparation sections.

The notebook currently uses Kaggle paths such as `/kaggle/input` and `/kaggle/working`, so it is an experiment notebook rather than a packaged application.

## Limitations and Responsible Use

- The O*NET data and the generated answers should not be treated as a final hiring, admissions, or career decision.
- Fairness scores depend on the selected counterfactual templates, sampled occupations, and definitions used in the notebook.
- Word-overlap faithfulness is a simple diagnostic and does not prove that an answer is factually correct.
- The notebook depends on a large language model and may require substantial memory and execution time.
- Reproducibility outside Kaggle requires the same dataset files, Python packages, model access, and updated file paths.

## Repository Contents

- [`ownrag-projectdone.ipynb`](ownrag-projectdone.ipynb): complete implementation, experiments, examples, and saved-result generation.
- `README.md`: technical project explanation and setup notes.
