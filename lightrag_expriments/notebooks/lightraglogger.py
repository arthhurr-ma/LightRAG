import os
import json
import logging
import time
import asyncio
import tiktoken
from dataclasses import asdict, dataclass
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete

class LoggingLightRAG(LightRAG):
    """A subclass of LightRAG with logging and writing capabilities."""

    def __init__(self, log_dir, *args, **kwargs):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_file = os.path.join(self.log_dir, "usage.log")
        self._set_logger(log_file)
        
        super().__init__(*args, **kwargs)
        self.api_call_count = 0
        self.total_token_usage = 0
        self._log_initialization()
        self.encoding = tiktoken.encoding_for_model(self.tiktoken_model_name)
    
    def _set_logger(self, log_file):
        """Configure the logger to write to a file."""
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logging.info(f"Logger initialized. Writing logs to {log_file}.")
    
    def _log_initialization(self):
        """Log the initialization details."""
        logging.info("LoggingLightRAG initialized with the following parameters:")
        try:
            attributes = asdict(self) if isinstance(self, dataclass) else vars(self)
            for attr, value in attributes.items():
                logging.info(f"- {attr}: {value}")

            # Write initialization details to a file
            init_file = os.path.join(self.log_dir, "initialization.json")
            with open(init_file, "w") as f:
                json.dump(attributes, f, indent=4)
            logging.info(f"Initialization details written to {init_file}.")
        except Exception as e:
            logging.error(f"Error logging initialization parameters: {e}")

    def _write_to_file(self, filename, data):
        """Write data to file."""
        try:
            filepath = os.path.join(self.log_dir, filename)
            with open(filepath, "a") as f:
                json.dump(data, f, indent=4)
                f.write("\n")
            logging.info(f"Data written to {filepath}.")
        except Exception as e:
            logging.error(f"Error writing data to file {filename}: {e}")

    def _log_api_usage(self, operation, response):
        """Log API usage, including operation type, token usage, and latency."""
        self.api_call_count += 1

        if 'usage' in response:
            token_usage = response['usage'].get('total_tokens', 0)
            self.total_token_usage += token_usage

            # Log API usage details
            logging.info(f"API Call #{self.api_call_count} ({operation})")
            logging.info(f"  Token Usage: {token_usage}")
            logging.info(f"  Total Token Usage: {self.total_token_usage}")

            # Log to file
            log_data = {
                "api_call": self.api_call_count,
                "operation": operation,
                "token_usage": token_usage,
                "total_token_usage": self.total_token_usage,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._write_to_file("api_usage_log.json", log_data)
        else:
            logging.warning(f"API Response for {operation} did not include 'usage' data.")

    async def insert(self, document):
        """Insert document asynchronously and log API usage."""
        try:
            loop = asyncio.get_event_loop()
            result = await self.ainsert(document)

            if result is None:
                logging.warning("Insert operation returned None.")
                return None

            # Log API usage (assuming result contains OpenAI API response)
            self._log_api_usage("insert", result)
            logging.info("Document(s) inserted into LoggingLightRAG.")
            return result
        except Exception as e:
            logging.error(f"Error during document insertion: {e}")
            raise

    async def query(self, query, param):
        """Query and log API usage."""
        try:
            start_time = time.time()
            response = await super().query(query, param)
            end_time = time.time()

            # Log API usage from OpenAI response
            self._log_api_usage("query", response)

            return response
        except Exception as e:
            logging.error(f"Error during query execution: {e}")
            raise
