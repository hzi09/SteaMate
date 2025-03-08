# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain.schema import Document
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv
from langchain_community.vectorstores import PGVector # pgvector용 모듈
import os
import pandas as pd
load_dotenv()

# API 키 환경변수에서 가져오기
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# PostgreSQL 연결 문자열 (환경 변수에서 가져오거나 직접 설정)
CONNECTION_STRING = os.getenv('DATABASE_URL')  # 예: "postgresql://user:password@localhost:5432/dbname"

# 챗봇 모델 설정
chat = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# 파서
str_outputparser = StrOutputParser()

# 템플릿
prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", "game : {context}"),
        (
            "system",
            """당신은 게임 추천 전문가입니다. 반드시 다음 언어로 답변하시오 : korean.
            - 아래 조건에 따라 답변하시오
            """,
        ),
        ("human", "{input}"),
    ]
)

# def load_csv():
#     file_path=os.path.abspath('chatmate/data/games.csv')
#     data = pd.read_csv(file_path, encoding="utf-8")
#     documents = [
#         Document(
#             page_content=" | ".join([f"{col}: {value}" for col, value in row.items() if col != "appid"]),  # AppID 제외하고 모든 컬럼을 결합
#             metadata={"appid": row["appid"], "genres": row["genres"]}  # 메타데이터에 AppID와 Category 포함
#         )
#         for _, row in data.iterrows()
#     ]
    
#     return documents

# def create_vectorstore(data):
#     try:
#         # PGVector를 사용한 벡터 스토어 생성
#         vector_store = PGVector.from_documents(
#             documents=data,
#             embedding=embeddings,
#             connection=CONNECTION_STRING,
#             collection_name="games_collection",  # 테이블 이름 역할
#         )
        
#         print("PGVector 벡터 DB를 생성하였습니다.")
#         # 우선 모든에러처리
#     except Exception as e: 
#         print(f"벡터 db 초기화 중 오류 :: {e}")
#     return vector_store

def load_and_chunk_csv(chunk_size=100):
    file_path = os.path.abspath('chatmate/data/games.csv')
    data = pd.read_csv(file_path, encoding="utf-8")
    
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        chunk_documents = [
            Document(
                page_content=" | ".join([f"{col}: {value}" for col, value in row.items() if col != "appid"]),
                metadata={"appid": row["appid"], "genres": row["genres"]}
            )
            for _, row in chunk.iterrows()
        ]
        chunks.append(chunk_documents)
    
    return chunks

def create_vectorstore_from_chunks(chunks):
    vector_store = None
    for chunk in chunks:
        if vector_store is None:
            vector_store = PGVector.from_documents(
                documents=chunk,
                embedding=embeddings,
                connection_string=CONNECTION_STRING,
                collection_name="games_collection",
                use_jsonb=True
            )
        else:
            vector_store.add_documents(chunk)
    
    return vector_store


def initialize_vectorstore():
    
    try: 
        # 벡터 스토어 로드 시도
        vector_store = PGVector(
            embedding_function=embeddings,
            connection_string=CONNECTION_STRING,
            collection_name="games_collection",
            use_jsonb=True
        )
        # 우선 모든에러처리
    except Exception as e: 
        print(f"벡터 db 초기화 중 오류 :: {e}")
    
    # 데이터 비어있는지 확인
    sample = vector_store.similarity_search("test", k=1)
    if not sample:
        print("PGVector 벡터 DB가 비어 있습니다. 데이터를 생성합니다.")
        data = load_and_chunk_csv()
        vector_store = create_vectorstore_from_chunks(data)
    else:
        print("기존 PGVector 벡터 DB를 로드했습니다.")
    
    return vector_store
# 벡터 스토어 초기화
vector_store = initialize_vectorstore()

# 벡터 DB에서 질문을 검색할 리트리버
retriever = vector_store.as_retriever()

def docs_join_logic(docs):
    return "\n".join([doc.page_content for doc in docs])

# 가져온 문서 붙이기
docs_join = RunnableLambda(docs_join_logic)

# 체인
rag_chain =  chat | str_outputparser | retriever | docs_join
chain = prompt | chat | str_outputparser

store = {}

# 대화 세션에 대화를 입력해줄 함수
def get_session_history(session_ids):
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]

# 체인을 묶어 기억해줄 객체
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def chatbot_call(user_input, session_id):
    # RAG 체인을 통해 컨텍스트 생성
    context = rag_chain.invoke(f"Translate the following question into English: {user_input}")
    
    answer = chain_with_history.invoke(
        {"input" : user_input, "context" : context},
        config ={"configurable": {"session_id": session_id}}
    )
    print(context)
    return answer