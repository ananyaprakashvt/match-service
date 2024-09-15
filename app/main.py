#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from flask import Flask, jsonify, request


app = Flask(__name__)

#%% loading data
data = pd.read-csv('user.csv')
print(data.head(10))








#%%
@app.route("/", methods=['GET', 'POST'])
def index():
    return  {
        "path" : request.path,
        "method" : request.method,
        "headers" : dict(request.headers),
        "args" : dict(request.args),
        "body" : request.data.decode('utf-8')
    }


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
