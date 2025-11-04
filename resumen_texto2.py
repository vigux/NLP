from transformers import pipeline

resumen = pipeline("summarization", model="facebook/bart-large-cnn")
texto = """La inteligencia artificial está transformando sectores como la medicina, la educación y la industria.
Permite optimizar procesos y crear soluciones innovadoras que mejoran la calidad de vida."""
print(resumen(texto, max_length=40, min_length=15, do_sample=False))
