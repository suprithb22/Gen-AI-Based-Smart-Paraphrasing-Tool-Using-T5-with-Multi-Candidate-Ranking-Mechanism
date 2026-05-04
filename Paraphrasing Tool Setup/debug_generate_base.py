from transformers import T5Tokenizer, T5ForConditionalGeneration

path = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(path)
model = T5ForConditionalGeneration.from_pretrained(path)
model.eval()

text = "paraphrase: Artificial intelligence is transforming the way we live."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(
    **inputs, 
    max_length=50, 
    num_beams=5, 
    num_return_sequences=5
)
for out in outputs:
    print("OUTPUT:", repr(tokenizer.decode(out, skip_special_tokens=True)))
