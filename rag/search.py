import os

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

def semantic_search(query, embedding_model, num_chunks=5):
    embedding_query = np.array(embedding_model.embed_query(query))
    with psycopg.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT *, (embedding <=> %s) AS similarity_score FROM document ORDER BY similarity_score LIMIT %s", (embedding_query, num_chunks))
            rows = cur.fetchall()
            semantic_context = [{"id": row[0], "text": row[1], "source": row[2], "score": row[4]} for row in rows]

    return semantic_context
