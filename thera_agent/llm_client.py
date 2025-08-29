import os, asyncio, json

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.v1 = None
        try:
            # v1.x client
            from openai import AsyncOpenAI  # type: ignore
            self.v1 = AsyncOpenAI(api_key=self.api_key)
            self.is_v1 = True
        except Exception:
            import openai  # v0.28
            openai.api_key = self.api_key
            self.openai_028 = openai
            self.is_v1 = False

    async def chat(self, model: str, messages, temperature=0.3, max_tokens=1500, timeout=30, response_format=None):
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        if self.is_v1:
            kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            if response_format and model in ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo-1106"]:
                kwargs["response_format"] = response_format
            coro = self.v1.chat.completions.create(**kwargs)
        else:
            coro = self.openai_028.ChatCompletion.acreate(model=model, messages=messages,
                                                          temperature=temperature, max_tokens=max_tokens)
        return await asyncio.wait_for(coro, timeout=timeout)

def extract_json(text: str) -> str:
    if not text: return "{}"
    t = text.strip()
    if t.startswith("```json"): t = t[7:]
    if t.startswith("```"): t = t[3:]
    if t.endswith("```"): t = t[:-3]
    return t.strip()
