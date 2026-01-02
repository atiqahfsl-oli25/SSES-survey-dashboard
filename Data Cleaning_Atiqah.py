import pandas as pd
#drop unnecessary column
df = df.drop('timestamp', axis=1)
df = df.drop('Email Address', axis=1)
df
