import os
import ray
import psycopg
from pgvector.psycopg import register_vector
from  functools import partial
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.config import EFS_DIR
from rag.data import extract_sections
from rag.embed import EmbedChunks, get_embedding_model
from rag.utils import execute_bash

from dotenv import load_dotenv; load_dotenv()

class StoreResults:
    def __call__(self, batch):
        with psycopg.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for text, source, embedding in zip(batch["text"], batch["source"], batch["embeddings"]):
                    cur.execute("INSERT INTO document (text, source, embedding) VALUES (%s, %s, %s)", (text, source, embedding,),)
        return {}


def chunk_section(section, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.create_documents(
        texts=[section["text"]],
        metadatas=[{"source": section["source"]}]
    )
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]

def build_index(docs_dir, chunk_size, chunk_overlap, embedding_model_name, sql_dump_fp):
    # docs -> sections -> chunks
    ds = ray.data.from_items([{"path": path} for path in docs_dir.rglob("*.html") if not path.is_dir()])
    section_ds = ds.flat_map(extract_sections)
    chunk_ds = section_ds.flat_map(partial(
        chunk_section,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ))

    #Embed Chunks
    embedded_chunks = chunk_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": embedding_model_name},
        batch_size=100,
        concurrency=2
    )

    # Index data
    embedded_chunks.map_batches(
        StoreResults,
        batch_size=128,
        concurrency=2,
    ).count()

    # Save to SQL dump
    execute_bash(f"""rm -rf {sql_dump_fp} &&
                mkdir -p $(dirname "{sql_dump_fp}") && touch {sql_dump_fp} &&
                pg_dump "${os.environ["DB_CONNECTION_STRING"]}" -c > {sql_dump_fp}""" )
    print("Updated the index!")
