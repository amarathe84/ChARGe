"""
Debug wrapper for vLLM client to inspect raw responses and verify reasoning content modification.
This client captures and logs the raw HTTP response from vLLM to verify that reasoning_content
is being copied into the main response content.
"""

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import CreateResult, RequestUsage
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
import json
from loguru import logger
from typing import Any, Dict, List, Optional, Union
import sys


class DebugVLLMClient(OpenAIChatCompletionClient):
    """
    Debug wrapper for vLLM client that intercepts and logs raw API responses.
    This helps verify that vLLM modifications (reasoning -> content copy) are working.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_responses = []
        self.call_count = 0
        
    async def create(self, messages, **kwargs):
        """Intercept API calls and log raw responses before processing"""
        
        self.call_count += 1
        call_id = self.call_count
        
        # Log the request
        logger.info("\n" + "╔" + "="*98 + "╗")
        logger.info(f"║ vLLM API CALL #{call_id:03d} - REQUEST" + " "*70 + "║")
        logger.info("╠" + "="*98 + "╣")
        logger.info(f"║ Model: {kwargs.get('model', 'N/A'):<91s} ║")
        logger.info(f"║ Messages: {len(messages)} message(s)" + " "*73 + "║")
        
        # Log extra_body parameters (includes reasoning_effort)
        if 'extra_body' in kwargs:
            logger.info("║ Extra body parameters:" + " "*75 + "║")
            for key, value in kwargs['extra_body'].items():
                logger.info(f"║   {key}: {value:<86s} ║")
        
        logger.info("╚" + "="*98 + "╝\n")
        
        # Call parent to get the processed response
        response = await super().create(messages, **kwargs)
        
        # Try to access the underlying OpenAI client to get raw response
        # The parent class stores an AsyncOpenAI client internally
        raw_completion = None
        if hasattr(self, '_client') and isinstance(self._client, AsyncOpenAI):
            # We already made the call via super(), so we can't re-call
            # Instead, we'll inspect the CreateResult object
            pass
        
        # Log the processed response
        logger.info("\n" + "╔" + "="*98 + "╗")
        logger.info(f"║ vLLM API CALL #{call_id:03d} - RESPONSE (CreateResult)" + " "*52 + "║")
        logger.info("╠" + "="*98 + "╣")
        
        # Log content
        if hasattr(response, 'content') and response.content:
            logger.info("║ CONTENT (Final Answer Channel):" + " "*65 + "║")
            logger.info("╟" + "-"*98 + "╢")
            
            content_str = str(response.content)
            if isinstance(response.content, list):
                for idx, item in enumerate(response.content):
                    logger.info(f"║ [{idx}] {str(item)[:91]:<91s} ║")
                    # Handle multi-line content
                    remaining = str(item)[91:]
                    while remaining:
                        logger.info(f"║     {remaining[:91]:<91s} ║")
                        remaining = remaining[91:]
            else:
                # Split long content into multiple lines
                for line in content_str.split('\n'):
                    while line:
                        logger.info(f"║ {line[:96]:<96s} ║")
                        line = line[96:]
        
        # Log reasoning/thought if present
        if hasattr(response, 'thought') and response.thought:
            logger.info("╟" + "-"*98 + "╢")
            logger.info("║ THOUGHT (Reasoning Channel):" + " "*68 + "║")
            logger.info("╟" + "-"*98 + "╢")
            
            thought_str = str(response.thought)
            for line in thought_str.split('\n'):
                while line:
                    logger.info(f"║ {line[:96]:<96s} ║")
                    line = line[96:]
        
        # Log usage statistics
        if hasattr(response, 'usage') and response.usage:
            logger.info("╟" + "-"*98 + "╢")
            logger.info("║ USAGE STATISTICS:" + " "*80 + "║")
            logger.info(f"║   Prompt tokens: {response.usage.prompt_tokens:<82d} ║")
            logger.info(f"║   Completion tokens: {response.usage.completion_tokens:<78d} ║")
            if hasattr(response.usage, 'total_tokens'):
                logger.info(f"║   Total tokens: {response.usage.total_tokens:<83d} ║")
        
        # Log finish reason
        if hasattr(response, 'finish_reason'):
            logger.info("╟" + "-"*98 + "╢")
            logger.info(f"║ Finish reason: {response.finish_reason:<82s} ║")
        
        logger.info("╚" + "="*98 + "╝\n")
        
        # Store for later inspection
        self.raw_responses.append({
            'call_id': call_id,
            'response': response,
            'has_content': hasattr(response, 'content') and bool(response.content),
            'has_thought': hasattr(response, 'thought') and bool(response.thought),
        })
        
        return response
    
    def get_response_summary(self):
        """Get a summary of all captured responses"""
        logger.info("\n" + "╔" + "="*98 + "╗")
        logger.info("║ VLLM RESPONSE SUMMARY" + " "*75 + "║")
        logger.info("╠" + "="*98 + "╣")
        logger.info(f"║ Total API calls: {self.call_count:<82d} ║")
        logger.info("╟" + "-"*98 + "╢")
        
        for idx, resp_info in enumerate(self.raw_responses, 1):
            logger.info(f"║ Call #{resp_info['call_id']:03d}:" + " "*87 + "║")
            logger.info(f"║   Has content: {resp_info['has_content']!s:<84s} ║")
            logger.info(f"║   Has thought: {resp_info['has_thought']!s:<84s} ║")
            if idx < len(self.raw_responses):
                logger.info("╟" + "-"*98 + "╢")
        
        logger.info("╚" + "="*98 + "╝\n")


class RawVLLMResponseCapture(OpenAIChatCompletionClient):
    """
    Alternative approach: Directly intercept the OpenAI AsyncClient to capture raw HTTP response.
    This gets the raw JSON before AutoGen processes it.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_json_responses = []
        self.call_count = 0
        
    async def create(self, messages, **kwargs):
        """Override to capture and log raw API response"""
        
        self.call_count += 1
        call_id = self.call_count
        
        logger.info("\n" + "="*100)
        logger.info(f"   vLLM API CALL #{call_id:03d} - Sending request...")
        logger.info("   " + "="*100)
        
        # Call parent to make the actual API request
        # The parent's create() method returns a CreateResult, but internally it gets
        # a ChatCompletion from the OpenAI API
        response = await super().create(messages, **kwargs)
        
        # The parent already called the API, so we can't intercept the raw response easily.
        # Instead, let's make our own parallel call to capture the raw response.
        # But that's wasteful. Better approach: access the last response from the client.
        
        # Actually, let's just log what we got from the parent
        logger.info("\n   RESPONSE RECEIVED (CreateResult)")
        logger.info("   " + "-"*100)
        
        # Check for content
        has_content = hasattr(response, 'content') and response.content
        
        if has_content:
            content_str = str(response.content)
            # Check if content is a list
            if isinstance(response.content, list):
                logger.info(f"   Content is a list with {len(response.content)} item(s)")
                for idx, item in enumerate(response.content):
                    item_str = str(item)
                    logger.info(f"   Content[{idx}] (type: {type(item).__name__}): {item_str}")
            else:
                logger.info(f"   Content length: {len(content_str)} chars")
                # Print content without extra newlines
                logger.info(f"      {content_str}")
        
        # Check for thought/reasoning
        has_thought = hasattr(response, 'thought') and response.thought
        
        if has_thought:
            thought_str = str(response.thought).replace('\n\n', ' ').replace('\n', ' ')
            logger.info(f"   Thought length: {len(thought_str)} chars")
            logger.info(f"   Thought preview: {thought_str[:200]}")
            if len(thought_str) > 200:
                logger.info(f"      ... ({len(thought_str) - 200} more chars)")
        
        # Check for usage
        if hasattr(response, 'usage') and response.usage:
            logger.info(f"   Usage: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
        
        # Check finish reason
        if hasattr(response, 'finish_reason'):
            logger.info(f"   Finish reason: {response.finish_reason}")
        
        logger.info("\n" + "="*100 + "\n")
        
        # Store response
        self.raw_json_responses.append({
            'call_id': call_id,
            'response': response,
            'has_content': has_content,
            'has_thought': has_thought,
        })
        
        return response
