import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

path = Path(r'C:/Users/Ali/Downloads/smart_healthcare_dataset.csv')
df = pd.read_csv(path)

# encode gender
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

# create the same health_risk_score
df['health_risk_score'] = 0.25 * df['age'] + 0.25 * df['bmi'] + 0.25 * df['cholesterol'] + 0.25 * df['blood_pressure']

X = df.drop(columns=['heart_disease', 'diabetes', 'stroke'])
y = df['heart_disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print('accuracy', accuracy_score(y_test, y_pred))
print('train_acc', model.score(X_train_scaled, y_train))
print('class counts train', y_train.value_counts().to_dict())
print('class counts test', y_test.value_counts().to_dict())
print('classification report:\n', classification_report(y_test, y_pred))
