# --- START OF REVISED utils/llm.py (Final Parser Attempt) ---

"""Helper functions for LLM"""

import json
import re
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel, ValidationError
from langchain_core.language_models.chat_models import BaseChatModel
from utils.progress import progress

T = TypeVar('T', bound=BaseModel)

# --- Keep call_llm function exactly the same as the previous version ---
# --- (The one with enhanced debugging prints) ---
def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory = None
) -> T:
    """
    Makes an LLM call with retry logic, handling structured output with fallbacks.
    Handles cases where .with_structured_output is not implemented.
    """
    from llm.models import get_model, get_model_info, ModelProvider

    model_info = get_model_info(model_name)
    llm: Optional[BaseChatModel] = get_model(model_name, model_provider)

    if llm is None:
        print(f"Error: Could not initialize model {model_name} from {model_provider}")
        return create_default_response(pydantic_model, default_factory)

    should_attempt_structured_output = not (model_info and not model_info.has_json_mode())
    use_native_structured_output = False
    structured_llm = llm

    if should_attempt_structured_output:
        try:
            structured_llm = llm.with_structured_output(
                pydantic_model,
                method="json_mode",
            )
            use_native_structured_output = True
        except NotImplementedError:
            use_native_structured_output = False
        except Exception as e:
            print(f"Warning: Error trying to bind structured output for {model_name}: {e}. Falling back.")
            use_native_structured_output = False

    last_exception = None
    content_str_for_error = ""

    for attempt in range(max_retries):
        try:
            result = structured_llm.invoke(prompt)
            print(f"\nDEBUG [{agent_name} Attempt {attempt+1}]: RAW LLM OUTPUT:\n---\n{result}\n---\n") # Keep debug print

            content_str = None
            if use_native_structured_output:
                if isinstance(result, pydantic_model):
                    return result
                else:
                    print(f"Warning: Expected Pydantic obj from structured output, got {type(result)}. Falling back to manual parse.")
                    content_str = getattr(result, 'content', None)
                    if not isinstance(content_str, str):
                         raise Exception("Unexpected output format from native structured output.")
            else:
                if hasattr(result, 'content') and isinstance(result.content, str):
                    content_str = result.content
                elif isinstance(result, str):
                    content_str = result

            if content_str is not None:
                content_str_for_error = content_str
                provider_enum_value = model_info.provider if model_info else None
                # Use the most robust parser
                return parse_llm_output_robust(content_str, pydantic_model)
            else:
                raise Exception(f"LLM output has no parsable string content: {result}")

        except (ValidationError, json.JSONDecodeError) as parse_error:
            last_exception = parse_error
            error_msg = f"Parse/Validation Error Attempt {attempt + 1}/{max_retries}: {parse_error}"
            print(f"DEBUG [{agent_name} Attempt {attempt+1}]: PARSE/VALIDATION ERROR: {error_msg}") # Keep debug print
            if agent_name: progress.update_status(agent_name, None, f"Parse Error - retry {attempt + 1}")

        except Exception as e:
            last_exception = e
            print(f"DEBUG [{agent_name} Attempt {attempt+1}]: CAUGHT EXCEPTION: {type(e).__name__}: {e}") # Keep debug print
            error_msg = f"LLM Call/Processing Error Attempt {attempt + 1}/{max_retries}: {e}"
            if agent_name: progress.update_status(agent_name, None, f"LLM Error - retry {attempt + 1}")

        if attempt == max_retries - 1:
             print(f"Error: LLM call or processing failed after {max_retries} attempts.")
             print(f"DEBUG [{agent_name}]: Last Exception: {type(last_exception).__name__}: {last_exception}")
             print(f"DEBUG [{agent_name}]: Last Content Processed:\n---\n{content_str_for_error[:500]}...\n---\n")
             return create_default_response(pydantic_model, default_factory)

    return create_default_response(pydantic_model, default_factory)


# --- Helper Functions (REPLACE parse_llm_output_manual with parse_llm_output_robust) ---

def clean_json_string(json_string: str) -> str:
    """Attempts to clean common LLM JSON output issues before parsing."""
    # Remove common prefixes/suffixes like ```json, ```
    json_string = re.sub(r'^```json\s*', '', json_string.strip())
    json_string = re.sub(r'```$', '', json_string.strip())
    json_string = re.sub(r'^```\s*', '', json_string.strip())

    # Attempt to remove extraneous characters before quoted keys/strings
    json_string = re.sub(r'(?<=[\s{\[,])\b[0-9a-fA-FxX]+([lLbuU]?)\b(?=\s*")', '', json_string)

    # Remove trailing commas before closing braces/brackets
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)

    # Remove potential control characters (except common whitespace \n, \t, \r)
    json_string = ''.join(c for c in json_string if c.isprintable() or c in '\n\t\r')

    return json_string.strip()


# --- NEW ROBUST PARSER ---
def parse_llm_output_robust(content: str, pydantic_model: Type[T]) -> T:
    """
    Attempts to parse JSON from LLM output, trying various extraction methods.
    """
    parsed_json = None
    cleaned_content_for_error = "" # Store what was attempted for error msg

    # Attempt 1: Extract from Markdown Block
    markdown_content = extract_string_from_markdown_robust(content)
    if markdown_content:
        try:
            cleaned_content = clean_json_string(markdown_content)
            cleaned_content_for_error = cleaned_content # Store for potential error msg
            parsed_json = json.loads(cleaned_content)
            print(f"DEBUG: Successfully parsed from Markdown block.") # Optional debug
            return pydantic_model(**parsed_json) # Validate and return
        except (json.JSONDecodeError, ValidationError, TypeError) as e_md:
            print(f"DEBUG: Failed parsing Markdown content ({type(e_md).__name__}). Trying other methods.")
            # Fall through if markdown parsing fails

    # Attempt 2: Find JSON Object within Full Content
    try:
        # Find the first '{' and the last '}' to isolate potential JSON
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json_str = content[first_brace : last_brace + 1]
            cleaned_content = clean_json_string(potential_json_str)
            cleaned_content_for_error = cleaned_content # Store for potential error msg
            parsed_json = json.loads(cleaned_content)
            print(f"DEBUG: Successfully parsed from bounded {{...}} block.") # Optional debug
            return pydantic_model(**parsed_json) # Validate and return
    except (json.JSONDecodeError, ValidationError, TypeError) as e_bound:
        print(f"DEBUG: Failed parsing bounded {{...}} content ({type(e_bound).__name__}). Trying full cleaned content.")
        # Fall through if bounded parsing fails

    # Attempt 3: Try parsing the entire cleaned content as JSON
    try:
        cleaned_content = clean_json_string(content)
        cleaned_content_for_error = cleaned_content # Store for potential error msg
        # Only attempt if it looks like it starts with JSON object/array
        if cleaned_content.startswith('{') or cleaned_content.startswith('['):
             parsed_json = json.loads(cleaned_content)
             print(f"DEBUG: Successfully parsed from full cleaned content.") # Optional debug
             return pydantic_model(**parsed_json) # Validate and return
        else:
             print(f"DEBUG: Full cleaned content does not start with {{ or [, skipping full parse.")
             raise Exception("Content does not appear to be JSON after cleaning.") # Raise specific error

    except (json.JSONDecodeError, ValidationError, TypeError) as e_full:
         # If all methods fail, raise the final exception
         raise Exception(f"Failed to find/parse valid JSON after all methods: {e_full}\nLast Cleaned Attempt: {cleaned_content_for_error[:200]}...")
    except Exception as e_other: # Catch the specific exception from the check above
         raise e_other # Re-raise it


# --- Use Robust Markdown Extractor ---
def extract_string_from_markdown_robust(content: str) -> Optional[str]:
    """Extracts content string embedded within general markdown code blocks (``` ... ```)."""
    # Regex to find ``` optionally followed by 'json', then capture content until ```
    # DOTALL allows '.' to match newlines, IGNORECASE handles 'json'/'JSON'
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() # Return the captured JSON string { ... }

    # Fallback for cases where JSON might not start immediately after ```
    # Look for ```, then find the first { after it, then find the last } before the closing ```
    match_block = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if match_block:
         block_content = match_block.group(1)
         first_brace = block_content.find('{')
         last_brace = block_content.rfind('}')
         if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
              return block_content[first_brace : last_brace + 1].strip()

    return None # Return None if no valid block found

# --- Keep create_default_response function exactly the same ---
def create_default_response(model_class: Type[T], default_factory=None) -> T:
    """Creates a safe default response based on the model's fields or uses factory."""
    if default_factory:
        try: return default_factory()
        except Exception as e: print(f"Error calling default_factory: {e}. Creating basic default.")
    default_values = {}
    try:
        try: from typing import Literal
        except ImportError: Literal = None
        for field_name, field in model_class.model_fields.items():
            annotation = field.annotation
            origin = getattr(annotation, "__origin__", None)
            if annotation == str: default_values[field_name] = "Error in analysis, using default"
            elif annotation == float: default_values[field_name] = 0.0
            elif annotation == int: default_values[field_name] = 0
            elif annotation == bool: default_values[field_name] = False
            elif origin == list: default_values[field_name] = []
            elif origin == dict: default_values[field_name] = {}
            elif Literal and origin == Literal:
                 args = getattr(annotation, "__args__", [])
                 if args: default_values[field_name] = args[0]
                 else: default_values[field_name] = None
            else: default_values[field_name] = None
        return model_class(**default_values)
    except Exception as e:
         print(f"Critical Error: Failed to create basic default response for {model_class.__name__}: {e}")
         return {"error": "Failed to generate default response"}

# --- END OF REVISED utils/llm.py (Final Parser Attempt) ---
