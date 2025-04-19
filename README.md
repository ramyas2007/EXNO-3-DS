## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
REG NO: 212224040268

NAME  : RAMYA S
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Encoding Data (2).csv")
df
```

![image](https://github.com/user-attachments/assets/14df4c46-e8eb-4568-b2b9-bdd5372e2cc1)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/user-attachments/assets/d594643b-432e-4a13-9ee3-3137d8cc2762)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/user-attachments/assets/209fd618-c0c7-4df7-bc33-da02dc8fa24e)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![image](https://github.com/user-attachments/assets/75685b86-9b43-4ab2-8b8a-e9cecfc3bb8d)
```
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/ab963ce4-6409-4613-aed5-11982a618028)
```
pd.get_dummies(df2,columns=['nom_0'])
```

![image](https://github.com/user-attachments/assets/a3aa31a3-4bd2-4ca6-b286-dc510012a0dc)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data (2).csv")
df
```

![image](https://github.com/user-attachments/assets/d50e4b09-4d93-4bde-b0d3-dc120f1d648d)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```

![image](https://github.com/user-attachments/assets/0bdcf517-f8a1-493a-b212-27b358d3fcda)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```

![image](https://github.com/user-attachments/assets/47a1877b-335e-4b20-b2f6-b2061b202f41)
```
df=pd.read_csv("/content/Data_to_Transform (1).csv")
df
```

![image](https://github.com/user-attachments/assets/eecae3d3-f702-4509-a299-81d55679fc04)
```
df.skew()
```

![image](https://github.com/user-attachments/assets/7f33daff-5205-461d-863b-99e020b6e3e7)
```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/6cf409fc-4d1f-43d0-a281-0e03952a18fe)
```
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/user-attachments/assets/2c52d9f2-b1e5-468b-b0e0-b12f198af95e)
```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/42de70f2-212a-4b4f-9858-60d5b98326c2)
```
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/ae39892e-3e2b-4ee0-b3b2-24db110feb55)
```
df["Higly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/b6fb2a0e-9cb8-41f9-88da-7237907eea38)
```
df.skew()
```

![image](https://github.com/user-attachments/assets/fc63590b-54eb-45c9-98ad-8bc107a226b6)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![image](https://github.com/user-attachments/assets/479b9460-7580-43a4-8f41-acb6ddf09fd0)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![image](https://github.com/user-attachments/assets/0e9510ab-5d7a-4b19-a33c-cb162a802489)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/3d2d9935-4a91-495b-86ae-65d90bd89099)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/41a3ff93-dd55-433d-ac11-073d71f78004)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/bbaea421-5e43-4cd8-87b8-e09eb9890eda)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/d8e1690b-0c60-4988-85f4-ae0077ec695e)
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/cb108d22-6b25-4506-b53a-56ea4b69049b)
```
dt=pd.read_csv("/content/titanic_dataset (2).csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/924b0b4c-fd51-47b3-8c8e-090e956f109f)
```
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/c2ba9068-a140-46d6-9041-aa88ea0080b7)

# RESULT:
Feature Encoding and Transformation process has been successfully performed using
the data set.

       
