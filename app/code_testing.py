from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#%%


app = Flask(__name__)


data = pd.read_csv('user.csv', index_col=False)
print(data.head(10))
print(data.columns)

#%%
# Multi-label binarization for multi-label fields
mlb_clubs = MultiLabelBinarizer()
mlb_languages = MultiLabelBinarizer()
mlb_music = MultiLabelBinarizer()
mlb_pronouns = MultiLabelBinarizer()
#%%
data_clubs = pd.DataFrame(mlb_clubs.fit_transform(data['vt_clubs']), columns=mlb_clubs.classes_)
data_languages = pd.DataFrame(mlb_languages.fit_transform(data['languages']), columns=mlb_languages.classes_)
data_music = pd.DataFrame(mlb_music.fit_transform(data['music_genre']), columns=mlb_music.classes_)
data_pronouns = pd.DataFrame(mlb_pronouns.fit_transform(data['pronouns']), columns=mlb_pronouns.classes_)
print(data_clubs.head(5))
print(data_languages.head(5))
print(data_music.head(5))
print(data_pronouns.head(5))

#%% Combine encoded multi-label data with numeric features
user_features = pd.concat([
    data[['rating', 'convo_score']],
    data_clubs, data_languages, data_music, data_pronouns
], axis=1)

#%% Step 1: Standardize the data
scaler = StandardScaler()
user_features_scaled = scaler.fit_transform(user_features)

#%% Determine optimal n_clusters
def find_optimal_clusters(data, max_k=10):
    best_n_clusters = 2
    best_score = -1
    for n_clusters in range(2, max_k+1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    return best_n_clusters

#%% Perform KMeans clustering with a predefined number of clusters (e.g., 3 clusters)
optimal_n_clusters = find_optimal_clusters(user_features_scaled)
print("Optimal_n_cluster", optimal_n_clusters)
kmeans = KMeans(n_clusters=7, random_state=42)
data['cluster'] = kmeans.fit_predict(user_features_scaled)

#%% Function to calculate similarity scores
def calculate_similarity(target_user_index, other_user_indices, data_scaled):
    target_user = data_scaled[target_user_index].reshape(1, -1)
    other_users = data_scaled[other_user_indices]
    distances = pairwise_distances(target_user, other_users).flatten()
    similarity_scores = 1 - (distances / np.max(distances))  # Invert distance to similarity score
    print(similarity_scores)
    return similarity_scores

#%%
def test_calculate_similarity():
    # Pick a single user (e.g., the first user in the DataFrame)
    target_user_index = 5  # Index of the target user

    # Pick four arbitrary users (e.g., the next four users in the DataFrame)
    other_user_indices = [7, 2, 9, 4]  # Indices of the other users

    # Calculate similarity scores
    similarity_scores = calculate_similarity(target_user_index, other_user_indices, user_features_scaled)

    # Create a list of similarity scores with user IDs
    result = []
    for i, user_idx in enumerate(other_user_indices):
        result.append({
            "user_id": int(data.iloc[user_idx]['user_id']),
            "similarity_score": round(float(similarity_scores[i] * 100), 2)
        })

    return result


#%% Visualize the clusters
def viz(target_user_index, other_user_indices):
    # Perform PCA for 2D visualization
    pca = PCA(n_components=2)
    user_features_2d = pca.fit_transform(user_features_scaled)

    # Plot the clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(user_features_2d[:, 0], user_features_2d[:, 1], c=data['cluster'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Cluster')

    # Highlight the users in the test method
    target_user_index = 5
    other_user_indices = [7, 2, 9, 4]

    # Plot the target user
    plt.scatter(user_features_2d[target_user_index, 0], user_features_2d[target_user_index, 1], color='red', label='Target User', edgecolor='black', s=100)

    # Plot the other users

    plt.scatter(user_features_2d[other_user_indices, 0], user_features_2d[other_user_indices, 1], color='blue', label='Matched Drivers', edgecolor='black', s=100)

    # Add labels and legend
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('User Clusters Visualization')
    plt.legend()
    plt.show()
#%% Example usage
similarity_test_result = test_calculate_similarity()
print(similarity_test_result)
viz(5, [7, 2, 9, 4])
# pca = PCA(n_components=2)
# user_features_2d = pca.fit_transform(user_features_scaled)
#
# # Plot the clusters
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(user_features_2d[:, 0], user_features_2d[:, 1], c=data['cluster'], cmap='viridis', alpha=0.5)
# plt.colorbar(scatter, label='Cluster')
#
# # Highlight the users in the test method
# target_user_index = 0
# other_user_indices = [1, 2, 3, 4]
#
# # Plot the target user
# plt.scatter(user_features_2d[target_user_index, 0], user_features_2d[target_user_index, 1], color='red', label='Target User', edgecolor='black', s=100)
#
# # Plot the other users
# for idx in other_user_indices:
#     plt.scatter(user_features_2d[idx, 0], user_features_2d[idx, 1], color='blue', label='Other Users', edgecolor='black', s=100)
#
# # Add labels and legend
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('User Clusters Visualization')
# plt.legend(['Target User', 'Other Users'])
# plt.show()




#%% Existing index route logic
# @app.route("/", methods=['GET', 'POST'])
# def index():
#     return {
#         "path": request.path,
#         "method": request.method,
#         "headers": dict(request.headers),
#         "args": dict(request.args),
#         "body": request.data.decode('utf-8')
#     }

# New route for similarity calculation
@app.route("/compatibility", methods=['POST'])
def get_similarity():
    request_data = request.json
    target_user_id = request_data['target_user_id']
    user_list = request_data['user_list']

    # Get the indices of the users
    target_user_index = data[data['user_id'] == target_user_id].index[0]
    other_user_indices = data[data['user_id'].isin(user_list)].index.tolist()

    # Calculate similarity
    similarity_scores = calculate_similarity(target_user_index, other_user_indices, user_features_scaled)
    # visualize
    viz(target_user_index, other_user_indices)
    # Create response data
    response_data = []
    for i, user_idx in enumerate(other_user_indices):
        response_data.append({
            "user_id": int(data.iloc[user_idx]['user_id']),
            "similarity_score": round(float(similarity_scores[i] * 100), 2)
        })

    # Sort by similarity score in descending order
    response_data = sorted(response_data, key=lambda x: x['similarity_score'], reverse=True)

    # Return JSON response
    return jsonify(response_data)

app.run(debug=True, host='0.0.0.0', port=8080)