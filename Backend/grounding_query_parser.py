from system_prompts.yolo_filter_rules import *
from system_prompts.prompts import llm_get_filters_grounding_prompt
import re
import json
from difflib import SequenceMatcher
import torch

def parse_with_llm(llm_model, llm_tokenizer, prompt):
    """Parse grounding prompt using Qwen2.5-1.5B-Instruct."""
    
    system = llm_get_filters_grounding_prompt

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Query: \"{prompt}\""}
    ]
    
    # Apply chat template
    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False,
            pad_token_id=llm_tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    response = llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Try to extract and validate JSON
    try:
        # Find JSON in response
        json_match = re.search(r'\{[^{}]+\}', response)
        if json_match:
            parsed = json.loads(json_match.group())
            return json.dumps(parsed)  # Return clean JSON
    except:
        pass
    
    return response


def parse_with_llm_structured(llm_model, llm_tokenizer, prompt):
    """Parse and return as Python dict for direct use in grounding pipeline."""
    result = parse_with_llm(llm_model, llm_tokenizer, prompt)
    try:
        return json.loads(result)
    except:
        return None

def parse_with_llm_to_query(llm_model, llm_tokenizer, prompt):
    """
    Parse prompt using LLM and convert to query format.
    Returns None if LLM parsing fails.
    """
    try:
        llm_result = parse_with_llm_structured(llm_model, llm_tokenizer, prompt)
        if llm_result and llm_result.get('class'):
            return {
                'class': llm_result.get('class'),
                'class_confidence': 0.9,
                'spatial': llm_result.get('spatial'),
                'size': llm_result.get('size'),
                'ordinal': llm_result.get('ordinal'),
                'reference': llm_result.get('reference'),
                'color': llm_result.get('color'),
                'original_prompt': prompt
            }
    except Exception as e:
        print(f"LLM parsing error: {e}")
    return None


def fuzzy_match_class(word, threshold=0.75):
    """
    Find best matching class using fuzzy string matching.
    Handles typos like "airplan" → "airplane", "vehicel" → "vehicle"
    """
    word = word.lower().strip()
    
    # First check exact synonym match
    if word in SYNONYM_TO_CLASS:
        return SYNONYM_TO_CLASS[word]
    
    # Then try fuzzy matching against all synonyms
    best_match = None
    best_score = 0
    
    for synonym, cls in SYNONYM_TO_CLASS.items():
        score = SequenceMatcher(None, word, synonym).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = cls
    
    return best_match

def normalize_text(text):
    """Normalize text for better matching."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle contractions
    text = text.replace("'s", "")
    text = text.replace("'", "")
    
    # Remove articles for matching (but keep original for context)
    return text

def extract_class_from_prompt(prompt):
    """
    Extract target class from prompt using multiple strategies.
    EXCLUDES words that are part of reference phrases (e.g., "near the harbor" → harbor is reference, not target)
    Returns: (class_name, confidence_score)
    """
    prompt_lower = normalize_text(prompt)
    
    # First, remove reference phrases to avoid matching reference as target
    # e.g., "ship closest to the harbor" → "ship closest to the" (harbor removed)
    ref_patterns = [
        r"near (?:the |a )?(\w+)",
        r"close to (?:the |a )?(\w+)",
        r"closest to (?:the |a )?(\w+)",
        r"next to (?:the |a )?(\w+)",
        r"beside (?:the |a )?(\w+)",
        r"by the (\w+)",  # More specific to avoid removing "by" alone
        r"adjacent to (?:the |a )?(\w+)",
        r"nearby (?:the |a )?(\w+)",
        r"nearest to (?:the |a )?(\w+)",
    ]
    
    # Find and remove reference words from the prompt for class extraction
    prompt_for_class = prompt_lower
    for pattern in ref_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            ref_word = match.group(1)
            # Remove the reference word from the prompt used for class extraction
            prompt_for_class = prompt_for_class.replace(match.group(0), " ")
    
    prompt_words = prompt_for_class.split()
    
    # Strategy 1: Direct class name match (in cleaned prompt)
    for cls in DIOR_CLASSES:
        cls_normalized = cls.replace("-", " ").replace("_", " ")
        if cls_normalized in prompt_for_class or cls in prompt_for_class:
            return cls, 1.0
    
    # Strategy 2: Synonym match (multi-word first, in cleaned prompt)
    for synonym, cls in sorted(SYNONYM_TO_CLASS.items(), key=lambda x: -len(x[0])):
        if synonym in prompt_for_class:
            return cls, 0.95
    
    # Strategy 3: Single word synonym match (in cleaned prompt)
    for word in prompt_words:
        if word in SYNONYM_TO_CLASS:
            return SYNONYM_TO_CLASS[word], 0.9
    
    # Strategy 4: Fuzzy matching for typos
    for word in prompt_words:
        if len(word) >= 4:  # Only fuzzy match longer words
            matched = fuzzy_match_class(word, threshold=0.8)
            if matched:
                return matched, 0.7
    
    # Strategy 5: Try bigrams (two-word combinations)
    for i in range(len(prompt_words) - 1):
        bigram = f"{prompt_words[i]} {prompt_words[i+1]}"
        if bigram in SYNONYM_TO_CLASS:
            return SYNONYM_TO_CLASS[bigram], 0.85
    
    return None, 0.0

def extract_spatial_from_prompt(prompt):
    prompt_lower = normalize_text(prompt)
    
    # 1. Replace synonyms with word boundaries
    for syn, replacement in SPATIAL_SYNONYMS.items():
        # strict replacement: " upper " -> " top "
        prompt_lower = re.sub(r'\b' + re.escape(syn) + r'\b', replacement, prompt_lower)
    
    # 2. Match regions (Longest phrases first)
    for region in sorted(SPATIAL_REGIONS.keys(), key=len, reverse=True):
        pattern = r'\b' + re.escape(region) + r'\b'
        if re.search(pattern, prompt_lower):
            return region
            
    return None

def extract_size_from_prompt(prompt):
    """
    Extract size filter from prompt using whole-word matching.
    Prevents false positives like 'big' in 'ambiguous'.
    """
    prompt_lower = normalize_text(prompt)
    
    # Sort keywords by length (descending)
    sorted_keywords = sorted(SIZE_KEYWORD_TO_FILTER.items(), key=lambda x: len(x[0]), reverse=True)
    
    for keyword, filter_name in sorted_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        
        if re.search(pattern, prompt_lower):
            return filter_name
            
    return None


def extract_ordinal_from_prompt(prompt):
    """Extract ordinal filter from prompt."""
    prompt_lower = normalize_text(prompt)
    
    for ordinal in ORDINAL_FILTERS:
        if ordinal in prompt_lower:
            return ordinal
    
    return None


def extract_color_from_prompt(prompt):
    """
    Extract color from prompt using whole-word matching.
    Prevents false positives like 'tan' in 'instances'.
    """
    prompt_lower = normalize_text(prompt)
    
    # Sort colors by length (descending) to prioritize longer matches 
    sorted_colors = sorted(BASE_COLORS, key=len, reverse=True)
    
    for color in sorted_colors:
        pattern = r'\b' + re.escape(color) + r'\b'
        
        if re.search(pattern, prompt_lower):
            return color
            
    return None


def extract_reference_from_prompt(prompt):
    """Extract reference object for relative positioning."""
    prompt_lower = normalize_text(prompt)
    
    ref_patterns = [
        r"near (?:the |a )?(\w+)",
        r"close to (?:the |a )?(\w+)",
        r"closest to (?:the |a )?(\w+)",
        r"next to (?:the |a )?(\w+)",
        r"beside (?:the |a )?(\w+)",
        r"by the (\w+)",  # More specific to avoid removing "by" alone
        r"adjacent to (?:the |a )?(\w+)",
        r"nearby (?:the |a )?(\w+)",
        r"nearest to (?:the |a )?(\w+)",
    ]
    
    for pattern in ref_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            ref_word = match.group(1)
            # Try to map to a class
            if ref_word in SYNONYM_TO_CLASS:
                return SYNONYM_TO_CLASS[ref_word]
            # Try fuzzy match
            matched = fuzzy_match_class(ref_word)
            if matched:
                return matched
    
    return None

def parse_grounding_prompt_rule_based(prompt):
    """
    ROBUST prompt parser with synonym support, fuzzy matching, and multi-strategy extraction.
    
    Handles:
    - Synonyms: "plane" → airplane, "boat" → ship, "car" → vehicle
    - Typos: "airplan" → airplane, "vehicel" → vehicle
    - Plurals: "planes" → airplane, "cars" → vehicle
    - Complex phrases: "the biggest red car near the port"
    
    Returns:
        dict with parsed query components
    """
    # Extract all components
    target_class, class_confidence = extract_class_from_prompt(prompt)
    spatial = extract_spatial_from_prompt(prompt)
    size_filter = extract_size_from_prompt(prompt)
    ordinal = extract_ordinal_from_prompt(prompt)
    color = extract_color_from_prompt(prompt)
    reference = extract_reference_from_prompt(prompt)
    
    return {
        'class': target_class,
        'class_confidence': class_confidence,
        'spatial': spatial,
        'size': size_filter,
        'ordinal': ordinal,
        'reference': reference,
        'color': color,
        'original_prompt': prompt,
        'pipeline': 'rule_based'
    }

def parse_grounding_prompt(prompt, llm_model, llm_tokenizer):
    query = parse_grounding_prompt_rule_based(prompt)
    
    if query.get('class'):
        filters = []
        if query.get('spatial'): filters.append(f"spatial={query['spatial']}")
        if query.get('size'): filters.append(f"size={query['size']}")
        if query.get('color'): filters.append(f"color={query['color']}")
        if query.get('ordinal'): filters.append(f"ordinal={query['ordinal']}")
        if query.get('reference'): filters.append(f"reference={query['reference']}")
        if filters:
            print(f"\nFilters: {', '.join(filters)}\n")
        # result['pipeline_used'] = 'rule_based → yolo'
        # result['filters'] = filters
    else:
        # ─────────────────────────────────────────────────────────
        # STEP 2: LLM Fallback (Qwen2.5)
        # ─────────────────────────────────────────────────────────
        print(f"No class found, trying LLM fallback...")
        
        query = parse_with_llm_to_query(llm_model, llm_tokenizer, prompt)
        print(f"LLM Answer {query}")
        
        if query and query.get('class'):
            # result['pipeline_used'] = 'llm_fallback → yolo'
            # result['filters'] = query
            query['pipeline'] = 'llm_fallback'
        else:
            # ─────────────────────────────────────────────────────────
            # STEP 3: GDINO+SAM Fallback (no YOLO)
            # ─────────────────────────────────────────────────────────
            # print(f"LLM also failed, using GDINO+SAM only...")
            # result['pipeline_used'] = 'gdino_sam_only'
            query = {'class': None, 'original_prompt': prompt, 'pipeline_used':'gdino_sam_only'}
    
    return query

