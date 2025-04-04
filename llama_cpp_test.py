import llama_cpp
model = llama_cpp.Llama(
    model_path="models/gemma-3-4b-it-Q2_K.gguf",
)
print(model("The quick brown fox jumps ", stop=["."])["choices"][0]["text"])