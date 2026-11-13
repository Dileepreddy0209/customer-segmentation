# Generates a small synthetic file with the SAME schema as Online Retail II, for code testing only.
import numpy as np, pandas as pd
rng=np.random.default_rng(0); rows=[]
start=pd.Timestamp("2009-12-01")
for cid in range(12000,14000):
    n=rng.geometric(0.25)
    for _ in range(n):
        inv=str(rng.integers(489000,581000)); d=start+pd.Timedelta(days=int(rng.integers(0,739)))
        for _ in range(rng.integers(1,8)):
            rows.append([inv,str(rng.integers(10000,99999)),"ITEM",int(rng.integers(1,24)),d,round(float(rng.gamma(2,2)),2),cid,"United Kingdom"])
df=pd.DataFrame(rows,columns=["Invoice","StockCode","Description","Quantity","InvoiceDate","Price","Customer ID","Country"])
df.loc[df.sample(frac=.02,random_state=1).index,"Invoice"]="C"+df.Invoice
df.loc[df.sample(frac=.1,random_state=2).index,"Customer ID"]=np.nan
df.to_csv("synthetic_test.csv",index=False)
