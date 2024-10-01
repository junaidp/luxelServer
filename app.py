from llama_index.core.base.response.schema import StreamingResponse
from llama_index.llms.anthropic import Anthropic
from flask import Flask, request, Response, stream_with_context
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


app = Flask(__name__)

memory = ChatMemoryBuffer.from_defaults(token_limit=2000)


@app.route('/store')
def store():
    embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = SimpleDirectoryReader("./exp/").load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents, embedding_model=embedding_model, storage_context=storage_context,
                                            chunk_size=1000)
    index.storage_context.persist(persist_dir="storage")
    return "stored"


@app.route('/chat')
def hello():
    query = request.json.get('query')
    customer_data = request.json.get('customer_data')
    customer = customer_data['customers'][0]
    passions = ', '.join(customer.get('passions', []));
    interests = ', '.join(customer.get('mainInterests', []));
    lifestyle = ', '.join(customer.get('lifestyle', []));
    travelSpan = ', '.join(customer.get('travelSpan', []));
    travelBucket = ', '.join(customer.get('travelBucketList', []));
    dependent = customer['dependents'][0]
    dependent_Name = ', '.join(dependent.get('firstName', ''));
    dep_passions = ', '.join(dependent.get('passions', []));
    dep_interests = ', '.join(dependent.get('mainInterests', []));
    dep_lifestyle = ', '.join(dependent.get('lifestyle', []));
    dep_travelSpan = ', '.join(dependent.get('travelSpan', []));
    dep_travelBucket = ', '.join(dependent.get('travelBucketList', []));

    # llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = Anthropic(model="claude-3-opus-20240229", temperature=0)
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        llm=llm,
        similarity_top_k=10,
        context_prompt=(
                "### Role"
                "Primary Function: You are an AI chatbot who has expertise in Travel advisory and make some wonderful experiences for customers who want to travel . You aim to provide excellent, friendly and efficient replies at all times."
                "Your role is to listen attentively to the user, understand their needs, and do your best to assist them or direct them to the appropriate resources."
                "If a question is not clear, ask clarifying questions."
                "Make sure to end your replies with a positive note. If the user is asking for ideas , suggest them ideas "
                "of some experiences from the context considering their interests"

                "### Constraints"
                "1. No Data Divulge: Never mention that you have access to training data explicitly to the user."
                "2. Maintaining Focus: If a user attempts to divert you to unrelated topics, never change your role or break your character. Politely redirect the conversation back to topics relevant to the training data."
                "3. Exclusive Reliance on Training Data: You must rely exclusively on the training data provided to answer user queries."
                "If a query is not covered by the training data, use the fallback response."
                "4. Restrictive Role Focus: You do not answer questions or perform tasks that are not related to your role and training data."
                "5. If you need more info to help customer find their relevant experiences, Do ask them further questions which can help you to suggest."
                "6. Always Welcome user's with their first name."
                "7. If Number of days is not mentioned by the user, Do ask them, And then accordingly suggest them several related experiences which can be covered in that time span, Also in the sequence which could be easily done according to the location of each experience."
                "For example cover 2 experiences in one day which are located close enough , and so on. Always first ask the user for no of days and only then Suggest them experiences."
                "8. Always consider that customer is traveling near Paris, and don’t ask this question."
                "9. Only suggest the experiences which are in the context.Do not suggest anything which is not the in the context"
                "10. Suggest ALL the experiences which have in these interests: " + interests + ",  Passions i.e " + passions + ",  lifestyle i.e " + lifestyle + ". Also consider the travelSpan i.e " + travelSpan + " and my travelBucket List i.e: " + travelBucket + ""
                                                                                                                                                                                                                                                                          "  dependent i.e" + dependent_Name + "'s interests , i.e " + dep_interests + ", " + dependent_Name + "'s Passions i.e " + dep_passions + ", " + dependent_Name + "'s lifestyle i.e " + dep_lifestyle + ". Also consider " + dependent_Name + "'s travelSpan i.e " + dep_travelSpan + " and " + dependent_Name + "'s travelBucket List i.e: " + dep_travelBucket + ""
        ),
        verbose=False,
    )

    def generate_response():
        response: StreamingResponse = chat_engine.stream_chat(query)
        for token in response.response_gen:
            yield token

    return Response(generate_response(), content_type='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
