from ddgs import DDGS

def ddg(keywords, region='wt-wt', max_results=5):
    with DDGS() as ddgs:
        return list(ddgs.text(keywords, region=region, max_results=max_results))

# 这样就能保持原来的调用方式
results = ddg("nihao")
print(results)