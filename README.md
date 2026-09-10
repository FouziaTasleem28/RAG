# FairGraphRAG

This project explores a question-answering system for occupational information.
It combines the O*NET career database with a graph of relationships between occupations, skills, knowledge, education, software, and tasks.

## What I did

- Loaded and cleaned occupational data from O*NET.
- Connected related career information into a searchable graph.
- Added semantic search so a question can be matched with relevant information.
- Used the retrieved information to support answers about occupations and career requirements.
- Included fairness-focused checks to make the answers more balanced and less dependent on a single type of evidence.
- Tested the workflow with example occupational questions and displayed the results.

## Main file

Open [`ownrag-projectdone.ipynb`](ownrag-projectdone.ipynb) to see the complete work, including the data preparation, graph creation, search process, and example answers.

## Data and running the notebook

The notebook was prepared in a Kaggle environment and expects the O*NET dataset at a Kaggle input path. To run it somewhere else, download the same O*NET dataset and update the `BASE_PATH` value near the beginning of the notebook.

The notebook installs the Python packages it needs in its first cell. Because it uses language models, embeddings, and a graph, the first run may take some time.

## Project outcome

The result is a prototype that brings together structured career data and natural-language questions. It is intended for exploration and demonstration, not as a final career or hiring decision system.
