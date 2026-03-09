import os
import base64
import time
from pathlib import Path
from typing import Optional, List, Dict, Union
from dotenv import load_dotenv
from openai import AzureOpenAI
import warnings

# Load environment variables
load_dotenv()

# Disable SSL warnings and verification for corporate environments
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
os.environ["REQUESTS_CA_BUNDLE"] = ""  # Disables SSL verification for requests library
os.environ["CURL_CA_BUNDLE"] = ""      # Alternative env var for SSL
os.environ["PYTHONHTTPSVERIFY"] = "0"  # Python-wide HTTPS verification disable

# Disable SSL verification at urllib3 level (used by requests)
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# LangSmith tracing configuration (optional)
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
if LANGSMITH_ENABLED:
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "invoice-auditor")

try:
    from langsmith import Client as LangSmithClient
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False


class AzureOpenAIClient:
    """Client for Azure OpenAI API calls with vision support and LangSmith tracing"""
    
    def __init__(self, enable_langsmith: Optional[bool] = None):
        """Initialize Azure OpenAI client from environment variables
        
        Args:
            enable_langsmith: Override LangSmith tracing. If None, uses LANGCHAIN_TRACING_V2 env var
        """
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        # LangSmith configuration
        self.langsmith_enabled = enable_langsmith if enable_langsmith is not None else LANGSMITH_ENABLED
        self.langsmith_client = None
        if self.langsmith_enabled and LANGSMITH_AVAILABLE:
            try:
                # Monkey-patch requests to disable SSL verification for LangSmith
                import requests
                original_request = requests.Session.request
                def patched_request(self, method, url, **kwargs):
                    kwargs.setdefault('verify', False)
                    return original_request(self, method, url, **kwargs)
                requests.Session.request = patched_request
                
                self.langsmith_client = LangSmithClient()
            except Exception as e:
                print(f"Warning: Failed to initialize LangSmith client: {e}")
                self.langsmith_enabled = False
        
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Missing Azure OpenAI credentials. Please set AZURE_OPENAI_ENDPOINT "
                "and AZURE_OPENAI_KEY in your .env file"
            )
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        # Initialize embedding client (separate endpoint/deployment)
        self.embedding_endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT")
        self.embedding_api_key = os.getenv("AZURE_EMBEDDING_KEY")
        self.embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        self.embedding_api_version = os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15")
        
        self._embedding_client = None
        if self.embedding_endpoint and self.embedding_api_key:
            self._embedding_client = AzureOpenAI(
                azure_endpoint=self.embedding_endpoint,
                api_key=self.embedding_api_key,
                api_version=self.embedding_api_version
            )
    
    def chat(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Send a text prompt to Azure OpenAI with LangSmith tracing
        
        Args:
            prompt: The user prompt/question
            system_message: Optional system message to set behavior
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            metadata: Optional metadata for LangSmith tracing
            
        Returns:
            The model's response as string
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            end_time = time.time()
            
            result = response.choices[0].message.content
            
            # Log to LangSmith if enabled
            if self.langsmith_enabled and self.langsmith_client:
                self._log_llm_call(
                    name="azure_openai_chat",
                    run_type="llm",
                    inputs={"prompt": prompt, "system_message": system_message},
                    outputs={
                        "generation": result,
                        "usage_metadata": {
                            "input_tokens": response.usage.prompt_tokens,
                            "output_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    },
                    start_time=start_time,
                    end_time=end_time,
                    metadata={
                        "model": self.deployment,
                        "temperature": temperature,
                        "latency_ms": (end_time - start_time) * 1000,
                        **(metadata or {})
                    }
                )
            
            return result
        except Exception as e:
            if self.langsmith_enabled and self.langsmith_client:
                self._log_llm_call(
                    name="azure_openai_chat",
                    run_type="llm",
                    inputs={"prompt": prompt},
                    outputs={"error": str(e)},
                    start_time=start_time,
                    end_time=time.time(),
                    metadata={
                        "error": True,
                        "error_type": type(e).__name__,
                        "model": self.deployment
                    }
                )
            raise
    
    def chat_with_image(
        self,
        prompt: str,
        image_path: Optional[Union[str, Path]] = None,
        image_base64: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Send a prompt with an image to Azure OpenAI (vision model) with LangSmith tracing
        
        Args:
            prompt: The user prompt/question about the image
            image_path: Path to image file (jpg, png, etc.)
            image_base64: Base64 encoded image (if already encoded)
            system_message: Optional system message
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            metadata: Optional metadata for LangSmith tracing
            
        Returns:
            The model's response as string
            
        Note:
            Provide either image_path OR image_base64, not both
        """
        # Process image
        if image_base64 is None and image_path is None:
            raise ValueError("Must provide either image_path or image_base64")
        
        if image_base64 is None:
            image_path = Path(image_path)
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            image_format = image_path.suffix[1:]  # Remove the dot
        else:
            image_format = "png"  # Default format
        
        # Build messages
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{image_base64}"
                    }
                }
            ]
        })
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            end_time = time.time()
            
            result = response.choices[0].message.content
            
            # Log to LangSmith if enabled
            if self.langsmith_enabled and self.langsmith_client:
                self._log_llm_call(
                    name="azure_openai_vision",
                    run_type="llm",
                    inputs={"prompt": prompt, "image_path": str(image_path) if image_path else None},
                    outputs={
                        "generation": result,
                        "usage_metadata": {
                            "input_tokens": response.usage.prompt_tokens,
                            "output_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    },
                    start_time=start_time,
                    end_time=end_time,
                    metadata={
                        "model": self.deployment,
                        "temperature": temperature,
                        "latency_ms": (end_time - start_time) * 1000,
                        "vision": True,
                        **(metadata or {})
                    }
                )
            
            return result
        except Exception as e:
            if self.langsmith_enabled and self.langsmith_client:
                self._log_llm_call(
                    name="azure_openai_vision",
                    run_type="llm",
                    inputs={"prompt": prompt},
                    outputs={"error": str(e)},
                    start_time=start_time,
                    end_time=time.time(),
                    metadata={
                        "error": True,
                        "error_type": type(e).__name__,
                        "model": self.deployment,
                        "vision": True
                    }
                )
            raise
    
    def chat_conversation(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Send a multi-turn conversation to Azure OpenAI with LangSmith tracing
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     Example: [
                         {"role": "system", "content": "You are helpful"},
                         {"role": "user", "content": "Hello"},
                         {"role": "assistant", "content": "Hi there!"},
                         {"role": "user", "content": "How are you?"}
                     ]
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            metadata: Optional metadata for LangSmith tracing
            
        Returns:
            The model's response as string
        """
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            end_time = time.time()
            
            result = response.choices[0].message.content
            
            # Log to LangSmith if enabled
            if self.langsmith_enabled and self.langsmith_client:
                self._log_llm_call(
                    name="azure_openai_conversation",
                    run_type="llm",
                    inputs={"messages": messages},
                    outputs={
                        "generation": result,
                        "usage_metadata": {
                            "input_tokens": response.usage.prompt_tokens,
                            "output_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    },
                    start_time=start_time,
                    end_time=end_time,
                    metadata={
                        "model": self.deployment,
                        "temperature": temperature,
                        "latency_ms": (end_time - start_time) * 1000,
                        "conversation_turns": len(messages),
                        **(metadata or {})
                    }
                )
            
            return result
        except Exception as e:
            if self.langsmith_enabled and self.langsmith_client:
                self._log_llm_call(
                    name="azure_openai_conversation",
                    run_type="llm",
                    inputs={"messages": messages},
                    outputs={"error": str(e)},
                    start_time=start_time,
                    end_time=time.time(),
                    metadata={
                        "error": True,
                        "error_type": type(e).__name__,
                        "model": self.deployment
                    }
                )
            raise
    
    def create_embeddings(
        self, 
        text: Union[str, List[str]],
        metadata: Optional[Dict] = None
    ) -> Union[List[float], List[List[float]]]:
        """
        Create embeddings for text using Azure OpenAI text-embedding-3-large with LangSmith tracing
        
        Args:
            text: Single text string or list of text strings
            metadata: Optional metadata for LangSmith tracing
            
        Returns:
            List[float] for single text, or List[List[float]] for multiple texts
            
        Raises:
            ValueError: If embedding client is not configured
        """
        if not self._embedding_client:
            raise ValueError(
                "Embedding client not configured. Please set AZURE_EMBEDDING_ENDPOINT "
                "and AZURE_EMBEDDING_KEY in your .env file"
            )
        
        is_single = isinstance(text, str)
        input_text = [text] if is_single else text
        
        start_time = time.time()
        try:
            response = self._embedding_client.embeddings.create(
                model=self.embedding_deployment,
                input=input_text
            )
            end_time = time.time()
            
            embeddings = [item.embedding for item in response.data]
            result = embeddings[0] if is_single else embeddings
            
            # Log to LangSmith if enabled
            if self.langsmith_enabled and self.langsmith_client:
                self._log_llm_call(
                    name="azure_openai_embeddings",
                    run_type="embedding",
                    inputs={"text": text if is_single else f"{len(input_text)} texts"},
                    outputs={
                        "embedding_dimensions": len(embeddings[0]),
                        "usage_metadata": {
                            "total_tokens": response.usage.total_tokens
                        }
                    },
                    start_time=start_time,
                    end_time=end_time,
                    metadata={
                        "model": self.embedding_deployment,
                        "num_texts": len(input_text),
                        "latency_ms": (end_time - start_time) * 1000,
                        **(metadata or {})
                    }
                )
            
            return result
        except Exception as e:
            if self.langsmith_enabled and self.langsmith_client:
                self._log_llm_call(
                    name="azure_openai_embeddings",
                    run_type="embedding",
                    inputs={"text": text if is_single else f"{len(input_text)} texts"},
                    outputs={"error": str(e)},
                    start_time=start_time,
                    end_time=time.time(),
                    metadata={
                        "error": True,
                        "error_type": type(e).__name__,
                        "model": self.embedding_deployment
                    }
                )
            raise
    
    def _log_llm_call(self, name: str, run_type: str, inputs: Dict, outputs: Dict, start_time: float, end_time: float, metadata: Dict):
        """
        Log LLM call to LangSmith
        
        Args:
            name: Name of the operation
            run_type: Type of run ("llm" or "embedding")
            inputs: Input dictionary
            outputs: Output dictionary
            start_time: Start timestamp (seconds since epoch)
            end_time: End timestamp (seconds since epoch)
            metadata: Metadata dictionary
        """
        try:
            if self.langsmith_client:
                import uuid
                from datetime import datetime, timezone
                
                # Generate unique run ID
                run_id = str(uuid.uuid4())
                
                # Convert unix timestamps to datetime objects
                start_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
                end_dt = datetime.fromtimestamp(end_time, tz=timezone.utc)
                
                # Check if error in outputs
                error = outputs.get("error")
                
                # Create and immediately complete the run
                self.langsmith_client.create_run(
                    id=run_id,
                    name=name,
                    run_type=run_type,
                    inputs=inputs,
                    outputs=outputs,
                    start_time=start_dt,
                    end_time=end_dt,
                    extra=metadata,
                    error=error
                )
        except Exception as e:
            # Silently fail - don't break the main operation
            pass


# Convenience functions for quick usage
def send_prompt(prompt: str, **kwargs) -> str:
    """
    Quick function to send a text prompt to Azure OpenAI
    
    Args:
        prompt: The prompt text
        **kwargs: Additional arguments (system_message, temperature, max_tokens)
        
    Returns:
        Model response
    """
    client = AzureOpenAIClient()
    return client.chat(prompt, **kwargs)


def send_prompt_with_image(prompt: str, image_path: str, **kwargs) -> str:
    """
    Quick function to send a prompt with image to Azure OpenAI
    
    Args:
        prompt: The prompt text
        image_path: Path to image file
        **kwargs: Additional arguments (system_message, temperature, max_tokens)
        
    Returns:
        Model response
    """
    client = AzureOpenAIClient()
    return client.chat_with_image(prompt, image_path=image_path, **kwargs)