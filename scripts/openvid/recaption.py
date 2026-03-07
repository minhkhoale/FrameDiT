import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
llm = LLM(model=MODEL_NAME)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=256,
)

SYSTEM_PROMPT = """You rewrite video captions into short training prompts for a text-to-video model.

Rules:
- Output valid JSON only.
- For each item, return one short caption.
- Keep only main subject, main action, and important scene.
- Use 3 to 8 words.
- lowercase only
- no punctuation
- do not invent details
"""

def normalize_caption(text: str, max_words: int = 8) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return " ".join(text.split()[:max_words])

def build_prompt(batch):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Rewrite the following captions.\n"
                "Return JSON list in this format:\n"
                '[{"id": 1, "short_caption": "dog running on grass"}]\n\n'
                f"Input:\n{json.dumps(batch, ensure_ascii=False)}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def recaption_batch(batch):
    prompt = build_prompt(batch)
    outputs = llm.generate([prompt], sampling_params)
    text = outputs[0].outputs[0].text.strip()

    data = json.loads(text)
    result = [
        {
            "id": item["id"],
            "short_caption": normalize_caption(item["short_caption"]),
        }
        for item in data
    ]
    print(result)

recaption_batch(
    [
        'The video features a man in glasses with his hands clasped together, contemplating something. In the background, there is a muscular man holding dumbbells, suggesting a theme of fitness or bodybuilding. The style of the video is a split-screen, with the man in glasses on the left and the muscular man on the right. The overall tone of the video is contemplative and introspective, with a focus on the contrast between the two men.',
        "The video features a man in a green polo shirt with a logo on the left chest. He is wearing glasses and has short, graying hair. The man is speaking and appears to be in a dark room with a black background. The style of the video is a straightforward interview or discussion, with the focus on the man and his speech. The lighting is subdued, highlighting the man's face and the logo on his shirt. The overall tone of the video is serious and professional."
    ]
)