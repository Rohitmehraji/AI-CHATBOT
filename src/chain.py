from langchain_core.runnables import RunnableParallel, RunnablePassthrough


def rag_chain(llm, parser, prompt, retriever):
    chain = (
        RunnableParallel(
            {
                "context": retriever,  # fetches docs from retriever
                "question": RunnablePassthrough(),  # passes user question directly
            }
        )
        | prompt
        | llm
        | parser
    )

    return chain
