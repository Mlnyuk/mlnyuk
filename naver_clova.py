import requests
import uuid
import asyncio


class NaverClovaLLM:
    def __init__(
        self,
        api_key: str = "nv-3804806aea7c42a09763bd714ec260d8KkWw",
        base_url: str = "https://clovastudio.stream.ntruss.com",
        app_type: str = "testapp",
        model_name: str = "HCX-003"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.app_type = app_type
        self.model_name = model_name
        self.url = f"{base_url}/{app_type}/v1/chat-completions/{model_name}"
        
    def _get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4())
        }
    
    def chat(self, message: str, max_tokens: int = 256, temperature: float = 0.7):
        import time
        print(f"🤖 [NaverClovaLLM] API 호출 시작: {message[:50]}...")
        start_time = time.time()
        
        payload = {
            "messages": [
                {"role": "user", "content": message}
            ],
            "maxTokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(self.url, headers=self._get_headers(), json=payload)
        
        api_time = time.time() - start_time
        print(f"⏱️ [NaverClovaLLM] API 호출 완료: {api_time:.3f}초, Status: {response.status_code}")
        
        return {
            "status_code": response.status_code,
            "data": response.json() if response.status_code == 200 else None,
            "error": response.text if response.status_code != 200 else None
        }
    
    async def ainvoke(self, message: str, **kwargs):
        """비동기 인터페이스 - LangChain 호환성을 위한 메소드"""
        max_tokens = kwargs.get('max_tokens', 256)
        temperature = kwargs.get('temperature', 0.7)
        
        # 비동기적으로 chat 메소드 실행
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.chat, message, max_tokens, temperature)
        
        # LangChain 형태로 응답 반환
        if result["status_code"] == 200 and result["data"]:
            try:
                content = result["data"]["result"]["message"]["content"]
                return content
            except (KeyError, TypeError):
                return f"API 응답 파싱 오류: {result['data']}"
        else:
            return f"API 호출 오류: {result['error']}"