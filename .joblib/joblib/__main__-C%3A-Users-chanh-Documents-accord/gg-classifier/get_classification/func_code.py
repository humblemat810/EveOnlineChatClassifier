# first line: 37
@memory.cache( ignore = ["llm"])
def get_classification(llm: ChatGoogleGenerativeAI, messages, response_model):
    response = llm.with_structured_output(response_model, include_raw=True).invoke(messages)
    return response
