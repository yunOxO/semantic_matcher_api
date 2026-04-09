from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()


emb_model_name = './qwen3-embedding-0.6b'
emb_model = SentenceTransformer(emb_model_name,
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# emb_model.to(device)
# emb_model.eval()


def get_embs(sentences: List[str]):
    embeddings = emb_model.encode(sentences, normalize_embeddings=True)
    return embeddings.tolist()


def get_emb(sentence: str):
    embeddings = emb_model.encode(sentence, normalize_embeddings=True)
    return embeddings.tolist()


class Message(BaseModel):
    text: str = '正文'


class EmbeddingInput(BaseModel):
    input: str | List[str]
    model: str


class EmbeddingData(BaseModel):
    object: str = 'embedding'
    index: int = 0
    embedding: List[float] = []

    class Config:
        arbitrary_types_allowed = True


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingOutput(BaseModel):
    object: str = "list"
    data: List[EmbeddingData] = []
    model: str = 'qwen3-embedding-0.6b'
    usage: EmbeddingUsage = None

    class Config:
        arbitrary_types_allowed = True


async def handler_str_input(input: str):
    token_count = len(input)
    output_embedding = get_emb(input)
    out_data = EmbeddingData()
    out_usage = EmbeddingUsage()
    out_usage.prompt_tokens = token_count
    out_usage.total_tokens = token_count
    out_data.embedding = [float(e) for e in output_embedding]
    output = EmbeddingOutput()
    output.data.append(out_data)
    output.usage = out_usage
    return output


async def handler_list_input(input: List[str]):
    out_usage = EmbeddingUsage()
    output = EmbeddingOutput()
    res = get_embs(input)
    total_chars = sum(len(s) for s in input)
    out_usage.prompt_tokens = total_chars
    out_usage.total_tokens = total_chars
    index = 0
    for single in res:
        out_data = EmbeddingData()
        out_data.index = index
        out_data.embedding = [float(e) for e in single]
        output.data.append(out_data)
        index += 1
    output.usage = out_usage
    return output


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/embeddings")
async def get_embedding(embedding_input: EmbeddingInput):
    text = embedding_input.input
    if isinstance(text, str):
        output = await handler_str_input(text)
    else:
        output = await handler_list_input(text)
    output.model = embedding_input.model
    return output


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)

# curl -X POST http://0.0.0.0:8800/embeddings -H "Content-Type: application/json" -d '{"input": "Hello world", "model": "qwen3-embedding-0.6B"}'
