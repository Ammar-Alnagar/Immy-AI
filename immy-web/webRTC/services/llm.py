from openai import AsyncOpenAI
import os
import time
from typing import AsyncGenerator, Dict, Optional
from datetime import datetime
from .logger import Logger

class LLMService:
    def __init__(self):
        self.logger = Logger("LLMService")
        self.api_key = self._validate_api_key()
        self.client = self._initialize_openai_client()
        self.role = "You are a helpful AI assistant that provides clear and concise answers."
        self.max_tokens = 250
        self.logger.info("LLMService initialized successfully", {
            "role": self.role,
            "max_tokens": self.max_tokens
        })

    def _validate_api_key(self) -> str:
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.logger.error("OPENAI_API_KEY environment variable is not set")
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.logger.debug("OpenAI API key validated")
            return api_key
        except Exception as e:
            self.logger.error("API key validation failed", {"error": str(e)})
            raise

    def _initialize_openai_client(self) -> AsyncOpenAI:
        try:
            client = AsyncOpenAI(api_key=self.api_key)
            self.logger.info("OpenAI client initialized successfully")
            return client
        except Exception as e:
            self.logger.error("Failed to initialize OpenAI client", {"error": str(e)})
            raise ValueError(f"Invalid OpenAI API key: {str(e)}")

    async def stream_answer(self, question: str, metrics: Dict[str, int]) -> AsyncGenerator[str, None]:
        """Stream answer using OpenAI LLM API."""
        start_time = datetime.now().timestamp()
        total_tokens = 0
        try:
            self.logger.debug("Starting LLM stream", {"question_length": len(question)})
            metrics['llmStart'] = round(time.perf_counter() * 1000)
            
            stream = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.role},
                    {"role": "user", "content": question}
                ],
                max_tokens=self.max_tokens,
                stream=True
            )
            
            current_chunk = ""
            token_count = 0
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    current_chunk += content
                    token_count += 1
                    total_tokens += 1
                    
                    # When we have enough tokens or hit sentence boundary, yield the chunk
                    if token_count >= 30 or (content in ['.', '!', '?'] and token_count > 20):
                        if current_chunk.strip():
                            self.logger.debug("Yielding chunk", {
                                "chunk_length": len(current_chunk),
                                "token_count": token_count
                            })
                            yield current_chunk
                        current_chunk = ""
                        token_count = 0
            
            # Yield any remaining text
            if current_chunk.strip():
                self.logger.debug("Yielding final chunk", {
                    "chunk_length": len(current_chunk),
                    "token_count": token_count
                })
                yield current_chunk
            
            metrics['llmEnd'] = round(time.perf_counter() * 1000)
            duration_ms = round((datetime.now().timestamp() - start_time) * 1000, 2)
            
            self.logger.api_call(
                "GPT-4 Stream",
                success=True,
                duration_ms=duration_ms,
                context={
                    "total_tokens": total_tokens,
                    "question_length": len(question)
                }
            )
            
        except Exception as e:
            duration_ms = round((datetime.now().timestamp() - start_time) * 1000, 2)
            self.logger.api_call(
                "GPT-4 Stream",
                success=False,
                duration_ms=duration_ms,
                context={"error": str(e)}
            )
            raise RuntimeError(f"Answer error: {str(e)}")
