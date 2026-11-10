from thinkingears_asr import ASREnhancer


enhancer = ASREnhancer(hotwords=["共振峰", "梅尔频谱"])
print(enhancer.enhance("测量供振峰"))
print(f"Lexicon: {enhancer.lexicon_count:,}")
