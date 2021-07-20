from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import Flask, request, render_template
import joblib as jb

app = Flask(__name__)
model = jb.load('hypothyroid_new.joblib')

X = [[72.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 30.0, 1, 0.6, 1, 15.0, 1, 1.48, 1, 10.0, 0]]
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(X)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predictHypothyroid():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(sc.transform(final_features))

    if prediction == 1:
        result = "You have a Symtom of Hypothyroid!"
    elif prediction == 0:
        result = "You don't have a Hypothyroid."
    output = result
    return render_template('index.html', predicted='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
