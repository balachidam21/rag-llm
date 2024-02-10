import time
from typing import Any
from rag.embed import get_embedding_model
from rag.utils import get_num_tokens, trim, get_client
from rag.search import semantic_search


def response_stream(chat_completion):
    for chunk in chat_completion:
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content

def prepare_response(chat_completion, stream):
    if stream:
        return response_stream(chat_completion)
    else:
        return chat_completion.choices[0].messages.content


def generate_response(
        llm, temperature=0.0, stream=True,
        system_content="", assistant_content="", user_content="",
        max_retries=1, retry_interval=60):
    """Generate an response from LLM"""
    retry_count = 0
    client = get_client(llm)
    messages = [{"role": role, "content": content } for role, content in [("system", system_content), ("assistant", assistant_content), ("user", user_content)] if content]
    while retry_count <= max_retries:
        try:
            chat_completion = client.chat.completions.create(
                model=llm,
                messages=messages,
                stream=stream,
                temperature=temperature,
            )
            return prepare_response(chat_completion, stream=stream)
        except Exception as e:
            print(f"Exception occured: {e}")
            time.sleep(retry_interval)
            retry_count += 1

    return ""


class QueryAgent:
    def __init__(self, embedding_model_name="thenlper/gte-base",
                 llm="gpt-3.5-turbo", temperature=0.0,
                 max_content_length=4096, system_content="", assistant_content="") -> None:
        self.embedding_model = get_embedding_model(
            embedding_model_name=embedding_model_name,
            model_kwargs={"device":"cpu"},
            encode_kwargs={"device": "cpu", "batch_size": 100}
        )

        # Context length (restrict input length to 50% of total context length)
        max_context_length = int(0.5*max_context_length)

        #LLM
        self.llm = llm
        self.temperature = temperature
        self.context_length = max_context_length - get_num_tokens(system_content + assistant_content)
        self.system_content = system_content
        self.assistant_content = assistant_content


    def __call__(self, query, num_chunks=5, stream=True) -> Any:
        # Get Sources and Context
        context_results = semantic_search(
            query=query,
            embedding_model=self.embedding_model,
            num_chunks=num_chunks
        )

        #Generate Response
        context = [item["text"] for item in context_results]
        sources = [item["source"] for item in context_results]
        user_content = f"query: {query}, context: {context}"

        answer = generate_response(
            llm=self.llm,
            temperature=self.temperature,
            stream=stream,
            system_content=self.system_content,
            user_content=trim(user_content, self.context_length)
        )

        #Result
        result  =  {
            "question": query,
            "sources": sources,
            "answer": answer,
            "llm": self.llm,
        }
        return result
