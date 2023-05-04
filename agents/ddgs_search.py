from duckduckgo_search import ddg

keywords = 'decarbonization actions for office buildings filetype:pdf'
results = ddg(keywords, region='us-en', max_results=20)

for i, result in enumerate(results):
    print(f"({i}) {result}")