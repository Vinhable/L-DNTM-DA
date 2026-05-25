import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import os
import json
from datetime import datetime
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI 
import queue 
import threading

class LLMGuider(nn.Module):
    """
    A module to guide the topic model's training using feedback from a Large Language Model 
    (NVIDIA API - DeepSeek/Llama).
    Implements an advanced list-wise ranking loss based on KL-Divergence.
    """
    def __init__(self,
                 lambda_contrastive: float,
                 krouter_model_name: str, 
                 llm_max_workers: int,
                 llm_contrastive_temperature: float,
                 llm_guidance_refresh_rate: int,
                 llm_top_k: int,
                 llm_history_length: int,
                 llm_max_retries: int,
                 llm_retry_delay: int,
                 num_times: int,
                 num_topic: int,
                 llm_batch_size: int,
                 log_path: str = "./llm_logs/"):
        super().__init__()
        
        self.lambda_contrastive = lambda_contrastive
        self.krouter_model_name = krouter_model_name
        self.llm_max_workers = llm_max_workers
        self.llm_contrastive_temperature = llm_contrastive_temperature
        self.llm_guidance_refresh_rate = llm_guidance_refresh_rate
        self.llm_top_k = llm_top_k
        self.llm_history_length = llm_history_length
        self.llm_max_retries = llm_max_retries
        self.llm_retry_delay = llm_retry_delay
        self.num_times = num_times
        self.num_topic = num_topic
        self.llm_batch_size = llm_batch_size
        
        self.clients = []
        self.client_queue = queue.Queue()
        self.executor = None

        self.log_path = log_path
        self.log_lock = threading.Lock()
        if self.lambda_contrastive > 0 and self.log_path:
            os.makedirs(self.log_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(self.log_path, f"llm_guidance_log_{timestamp}.jsonl")

        if self.lambda_contrastive > 0:
            self._initialize_krouter_clients()
        
        self.guidance_cache = {}
        self.refined_top_words_cache = {}
        self.epoch_last_updated = -1

    def _initialize_krouter_clients(self):
        try:
            # Lấy KROUTER_API_KEYS từ biến môi trường
            api_keys_str = os.environ.get("KROUTER_API_KEYS")
            if not api_keys_str:
                print("FATAL: KROUTER_API_KEYS not found in environment. LLMGuider will be disabled.")
                return
            
            api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
            base_url = "https://api.krouter.net/v1"  # Base URL của KRouter
            
            print(f"Found {len(api_keys)} KRouter API key(s). Setting up {self.llm_max_workers} concurrent workers...")
            
            # Khởi tạo Queue với số lượng Clients bằng đúng số luồng (workers)
            for i in range(self.llm_max_workers):
                # Chia bài (round-robin) các keys cho các worker
                key_to_use = api_keys[i % len(api_keys)]
                client = OpenAI(base_url=base_url, api_key=key_to_use)
                self.client_queue.put(client)
            
            self.executor = ThreadPoolExecutor(max_workers=self.llm_max_workers)
            print(f"✅ KRouter Relay Ready: {self.llm_max_workers} Workers are standing by.")
            
        except Exception as e:
            print(f"Error initializing KRouter clients: {e}. LLMGuider disabled.")
    
    def _create_batch_prompt(self, topics_in_batch: list) -> str:
        topics_str = ""
        for topic in topics_in_batch:
            historical_str = "\n".join(
                [f"      - Time t-{i+1}: {', '.join(words)}" for i, words in enumerate(topic['historical_words'])]
            ) if topic['historical_words'] else "      - No historical data available."
            candidates_str = '", "'.join([c.replace('"', '\\"') for c in topic['current_words']])
            topics_str += f"""
  {{
    "topic_index": {topic['id'][1]},
    "history": [
{historical_str}
    ],
    "candidates": ["{candidates_str}"]
  }},"""
        topics_str = topics_str.strip()[:-1]

        return f"""
        You are a meticulous and discerning topic model expert. Your goal is to maximize Topic Coherence and Topic Diversity. 
        You will analyze a batch of topics. For each topic, refine its "candidates" list based on its "history".

        Here is the batch of topics to analyze:
        [
        {topics_str}
        ]

        *** CRITICAL RULES FOR VOCABULARY (MUST OBEY) ***
        1. NO HALLUCINATION: You are STRICTLY FORBIDDEN from inventing, suggesting, or generating any new words. 
        2. EXACT MATCH ONLY: You must ONLY evaluate words that are explicitly provided in the "candidates" list. 
        3. NO CORRECTIONS: Do NOT correct spelling mistakes. Do NOT change singular/plural forms. Output the word EXACTLY as it appears in the candidates list.

        *Your Step-by-Step Task for EACH topic:*
        1. *Identify Core Identity:* In one sentence, summarize the true, distinct semantic core of this topic based on its history.
        2. *Purge Noise:* Discard words from the "candidates" list that are typos, nonsensical, or overly generic (e.g., "people", "time").
        3. *Score Remaining Candidates:* For the surviving keywords, assign a "novelty_score" from 0.0 to 1.0. Differentiate the scores:
            - 0.8 - 1.0: Core, highly specific words defining the topic.
            - 0.5 - 0.7: Contextual, supporting words.
            - 0.2 - 0.4: Blurry edges, somewhat generic.
            - 0.0: Drift or completely off-topic noise.
        4. *Format Output:* Return ONLY a single, valid JSON object. No markdown.

        Example JSON structure:
        {{
          "analyzed_topics": [
            {{
              "topic_index": 0,
              "reasoning_summary": "Focuses on AI evolution.",
              "ranked_candidates": [
                {{ "word": "exactly_matching_word_1", "novelty_score": 1.0 }},
                {{ "word": "exactly_matching_word_2", "novelty_score": 0.6 }},
                {{ "word": "exactly_matching_word_3", "novelty_score": 0.2 }}
              ]
            }}
          ]
        }}
        """

    def _parse_batch_output(self, text: str) -> dict | None:
        try:
            # Xóa các thẻ markdown nếu model chatty
            clean_text = text.replace("```json", "").replace("```", "").strip()
            
            start_index = clean_text.find('{')
            end_index = clean_text.rfind('}')
            
            if start_index == -1 or end_index == -1 or end_index < start_index:
                print("\n[PARSE ERROR] Không tìm thấy JSON. RAW LLM OUTPUT (500 chars):")
                print("-" * 40)
                print(text[:500])
                print("-" * 40)
                return None

            json_str = clean_text[start_index : end_index + 1]
            data = json.loads(json_str)
            
            if "analyzed_topics" in data and isinstance(data["analyzed_topics"], list):
                results_dict = {
                    item['topic_index']: {
                        'reasoning_summary': item.get('reasoning_summary', 'N/A'),
                        'ranked_candidates': item.get('ranked_candidates', [])
                    } for item in data['analyzed_topics'] if 'topic_index' in item
                }
                return results_dict
            else:
                print(f"\n[PARSE ERROR] JSON không chứa 'analyzed_topics'. Các keys hiện có: {list(data.keys())}")
                return None
                
        except json.JSONDecodeError as e:
            print(f"\n[PARSE ERROR] Lỗi Decode JSON: {e}")
            print(f"RAW TEXT: {text[:500]}")
            return None
        except Exception as e:
            print(f"\n[PARSE ERROR] Lỗi không xác định: {e}")
            return None

    def _get_guidance_from_api_batched(self, batch_of_topics: list, current_time: int) -> dict:
        if not self.executor or not batch_of_topics: return {}
        
        prompt = self._create_batch_prompt(batch_of_topics)

        for attempt in range(self.llm_max_retries):
            # Rút Client khỏi Queue
            current_client = self.client_queue.get()
            
            try:
                # Phanh nhẹ 0.5s để hệ thống Relay của KRouter xử lý mượt mà hơn
                time.sleep(0.5) 
                
                start_time = time.time()
                
                completion = current_client.chat.completions.create(
                    model=self.krouter_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, 
                    top_p=1.0,
                    max_tokens=4096,
                    stream=False,
                    timeout=60.0,
                    extra_body={"reasoning_effort": "low"}
                )

                duration = time.time() - start_time
                response_text = completion.choices[0].message.content
                parsed_result = self._parse_batch_output(response_text)

                if parsed_result is not None: 
                    parsed_result['__meta__'] = {'duration': duration}

                    if hasattr(self, 'log_file') and self.log_file:
                        with self.log_lock: 
                            with open(self.log_file, "a", encoding="utf-8") as f:
                                log_entry = {
                                    "time_slice": current_time,
                                    "duration": duration,
                                    "raw_response": response_text 
                                }
                                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                                
                    return parsed_result
                
            except Exception as e:
                duration = time.time() - start_time
                error_msg = str(e)
                print(f"   -> [KRouter Error] t={current_time}, attempt {attempt+1}/{self.llm_max_retries} failed ({duration:.1f}s): {error_msg[:100]}")
                
                if attempt < self.llm_max_retries - 1:
                    if "429" in error_msg:
                        # KRouter xử lý Load Balancing tốt nên nếu gặp 429, thường chỉ cần nghỉ ngắn
                        import random
                        time.sleep(random.uniform(5.0, 10.0))
                    else:
                        time.sleep(self.llm_retry_delay)
            
            finally:
                # Trả lại Client vào Queue
                self.client_queue.put(current_client)
                    
        return {}

    def _chunk_list(self, lst, n):
        """Splits a list into sublists of size n."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
 
    def update_guidance_cache(self, current_epoch: int, beta: torch.Tensor, idx_to_word: dict):
        if not self.executor or self.llm_guidance_refresh_rate <= 0 or current_epoch % self.llm_guidance_refresh_rate != 0 or current_epoch == self.epoch_last_updated:
            return

        print(f"\n--- Epoch {current_epoch}: Refreshing LLM Guidance via KRouter ---")

        valid_vocab = set(idx_to_word.values())
        all_futures = []
        future_to_info = {}

        for t in range(1, self.num_times):
            all_topics_for_slice = []
            for k in range(self.num_topic):
                beta_k_t = beta[t, k, :]
                _, top_indices = torch.topk(beta_k_t, self.llm_top_k)
                current_words = [idx_to_word[idx.item()] for idx in top_indices]
                
                historical_words = []
                for hist_t in range(max(0, t - self.llm_history_length), t):
                    beta_k_hist_t = beta[hist_t, k, :]
                    _, top_indices_hist = torch.topk(beta_k_hist_t, self.llm_top_k)
                    historical_words.append([idx_to_word[idx.item()] for idx in top_indices_hist])
                
                all_topics_for_slice.append({
                    "id": (t, k), 
                    "current_words": current_words, 
                    "historical_words": historical_words
                })
            
            mini_batches = list(self._chunk_list(all_topics_for_slice, self.llm_batch_size))
            
            for batch in mini_batches:
                future = self.executor.submit(self._get_guidance_from_api_batched, batch, t)
                all_futures.append(future)
                future_to_info[future] = t

        with tqdm(total=len(all_futures), desc="Processing Batches") as pbar:
            for future in as_completed(all_futures):
                t = future_to_info[future]
                guidance_results_dict = future.result()
                
                for k_result, result_data in guidance_results_dict.items():
                    if k_result == '__meta__': continue
                    
                    topic_id = (t, k_result)
                    ranked_candidates = result_data.get('ranked_candidates', [])
                    self.guidance_cache[topic_id] = ranked_candidates
                    
                    try:
                        sorted_candidates = sorted(
                            ranked_candidates, 
                            key=lambda x: x.get("novelty_score", 0.0), 
                            reverse=True
                        )
                        clean_words = []
                        
                        for item in sorted_candidates:
                            raw_word = item.get("word", "")
                            if isinstance(raw_word, str) and raw_word.strip():
                                clean_word = raw_word.strip().lower()
                                if clean_word in valid_vocab and clean_word not in clean_words:
                                    clean_words.append(clean_word)
                                
                            if len(clean_words) >= 15:
                                break
                        if len(clean_words) < 15:
                            beta_k_t = beta[t, k_result, :]
                            _, top_indices = torch.topk(beta_k_t, 30) 
                            original_words = [idx_to_word[idx.item()] for idx in top_indices]
                            for orig_word in original_words:
                                if len(clean_words) >= 15:
                                    break
                                if orig_word not in clean_words:
                                    clean_words.append(orig_word)
                        self.refined_top_words_cache[topic_id] = clean_words
                        
                    except Exception as e:
                        print(f"   [Cache Error] Topic ({t}, {k_result}): {e}")
                pbar.update(1)
        
        self.epoch_last_updated = current_epoch

    def calculate_contrastive_loss(self, topic_embeddings: torch.Tensor, word_embeddings: torch.Tensor, word_to_idx: dict) -> torch.Tensor:
        """Calculate the list-wise ranking loss."""
        # 1. Kiểm tra Cache và Lambda
        if self.lambda_contrastive <= 0:
            return torch.tensor(0.0, device=topic_embeddings.device)
            
        if not self.guidance_cache:
            print("   [LLM Loss Debug] WARNING: guidance_cache is empty! Loss is 0.0")
            return torch.tensor(0.0, device=topic_embeddings.device)

        # 2. Kiểm tra word_to_idx
        if not word_to_idx:
            print("   [LLM Loss Debug] WARNING: word_to_idx dictionary is empty or None!")
            return torch.tensor(0.0, device=topic_embeddings.device)
            
        total_loss = 0.0
        num_guided_topics = 0
        
        for (t, k), ranked_candidates in self.guidance_cache.items():
            if not ranked_candidates or len(ranked_candidates) < 2:
                continue

            topic_emb = topic_embeddings[t, k]
            
            valid_words_embs = []
            target_scores = []
            for cand in ranked_candidates:
                raw_word = cand.get("word", "")
                if not isinstance(raw_word, str):
                    continue
                    
                clean_word = raw_word.strip().lower() 
                if clean_word in word_to_idx:
                    valid_words_embs.append(word_embeddings[word_to_idx[clean_word]])
                    
                    # Đảm bảo điểm số là số thực (float)
                    try:
                        score = float(cand.get("novelty_score", 0.0))
                    except (ValueError, TypeError):
                        score = 0.0
                    target_scores.append(score)
                    
            if len(valid_words_embs) < 2:
                print(f"   [LLM Loss Debug] Topic (t={t}, k={k}) skipped: Only {len(valid_words_embs)} valid words found in vocab.")
                continue
            
            candidate_embs = torch.stack(valid_words_embs)
            model_sims = F.cosine_similarity(topic_emb.unsqueeze(0), candidate_embs)
            raw_targets = torch.tensor(target_scores, device=topic_emb.device)
            target_sims = (raw_targets * 2.0) - 1.0 
            topic_loss = F.mse_loss(model_sims, target_sims, reduction='mean')
            
            total_loss += topic_loss
            num_guided_topics += 1

        if num_guided_topics == 0:
            print("   [LLM Loss Debug] FATAL: 0 topics were guided! All skipped due to vocab mismatch.")
            return torch.tensor(0.0, device=topic_embeddings.device)

        return total_loss / num_guided_topics