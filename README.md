# Conversational AI with Retrieval Augmented Generation

You can look at the [`notebooks`](https://github.com/balachidam21/rag-llm/tree/main/notebooks) folder to start reading and understand the project. The `rag` folder has the functions mentioned in the notebooks in modularized format for downstream usecases.

The data and vectorDB SQL dump is not added to the GitHub repo. To download the data, see the [vector_db_creation.ipynb](https://github.com/balachidam21/rag-llm/blob/main/notebooks/vector_db_creation.ipynb) and check the `Download Data` and `Index Data` section.


We have few env variables setup that are used in this project. The name of the environment variables are:
- OPENAI_API_BASE="https://api.openai.com/v1"
- OPEN_AI_KEY=""
- ANYSCALE_API_BASE="https://api.endpoints.anyscale.com/v1"
- ANYSCALE_API_KEY=""
- DB_CONNECTION_STRING="dbname=postgres user=postgres host=localhost password=postgres"
