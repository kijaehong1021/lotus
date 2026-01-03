import pandas as pd

import lotus
from lotus.models import LM
from lotus.cache import CacheFactory, CacheConfig, CacheType

cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=10000000, cache_dir=os.path.expanduser("~/.lotus/cache"))
cache = CacheFactory.create_cache(cache_config)
lm = LM(model="gpt-4o-mini", cache=cache)

lotus.settings.configure(lm=lm, enable_cache=True)
data = {
    "Course Name": [
        "History of the Atlantic World",
        "Riemannian Geometry",
        "Operating Systems",
        "Food Science",
        "Compilers",
        "Intro to computer science",
    ]
}

data2 = {"Skill": ["Math", "Computer Science"]}

df1 = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
join_instruction = "Taking {Course Name:left} will help me learn {Skill:right}"
res = df1.sem_join(df2, join_instruction)
print(res)
