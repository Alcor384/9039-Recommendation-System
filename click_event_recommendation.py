import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import faiss

# 读取数据
df = pd.read_csv("recipe_data.csv")

# 处理菜系类别（One-Hot Encoding）
cuisine_encoder = OneHotEncoder(sparse_output=False)
cuisine_encoded = cuisine_encoder.fit_transform(df[['Cuisine_Type']])

# 处理食材列表（TF-IDF）
tfidf_ingredients = TfidfVectorizer()
ingredients_matrix = tfidf_ingredients.fit_transform(df['Ingredients_List'])

# 处理烹饪步骤（TF-IDF）
tfidf_steps = TfidfVectorizer()
steps_matrix = tfidf_steps.fit_transform(df['Preparation_Steps'])

# 归一化数值特征
scaler = MinMaxScaler()
numeric_features = df[['Cooking_Time_Minutes', 'Calories_Per_Serving', 'Cost_Per_Serving', 'Popularity_Score']]
numeric_scaled = scaler.fit_transform(numeric_features)

# 组合所有特征
recipe_features = np.hstack((cuisine_encoded, ingredients_matrix.toarray(), steps_matrix.toarray(), numeric_scaled))

# 计算余弦相似度
similarity_matrix = cosine_similarity(recipe_features)

# 找到每个菜谱最相似的 K 个
K = 5
similar_recipes = np.argsort(-similarity_matrix, axis=1)[:, 1:K+1]

# FAISS 近似最近邻优化（使用 GPU）
d = recipe_features.shape[1]  # 特征向量维度
res = faiss.StandardGpuResources()  # 初始化 GPU 资源
index = faiss.IndexFlatL2(d)  # 创建 L2 距离索引
index = faiss.index_cpu_to_gpu(res, 0, index)  # 传输到 GPU
index.add(recipe_features.astype(np.float32))  # 添加所有菜谱特征

# 查询最相似的 K 个菜谱
_, indices = index.search(recipe_features.astype(np.float32), K+1)
similar_recipes_faiss = indices[:, 1:K+1]

# 获取推荐的菜谱
def get_recommendations(recipe_id, df, similar_recipes):
    recipe_index = df.index[df['Recipe_ID'] == recipe_id].tolist()[0]
    recommended_indices = similar_recipes[recipe_index]
    return df.iloc[recommended_indices][['Recipe_ID', 'Recipe_Name', 'Cuisine_Type']]

# 计算推荐的平均相似度
def compute_avg_similarity(recipe_id, similarity_matrix, df, similar_recipes):
    recipe_index = df.index[df['Recipe_ID'] == recipe_id].tolist()[0]
    recommended_indices = similar_recipes[recipe_index]
    similarities = similarity_matrix[recipe_index, recommended_indices]
    return np.mean(similarities), similarities

# 可视化推荐结果的相似度
def plot_similarity_distribution(recipe_id, similarity_matrix, df, similar_recipes):
    _, similarities = compute_avg_similarity(recipe_id, similarity_matrix, df, similar_recipes)
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(similarities)), similarities, color='skyblue')
    plt.xlabel("Recommended Recipe Rank")
    plt.ylabel("Cosine Similarity")
    plt.title(f"Similarity Distribution for Recipe {recipe_id}")
    plt.ylim(0, 1)
    plt.show()


clicked_recipe = "R002"
recommended_recipes = get_recommendations(clicked_recipe, df, similar_recipes_faiss)

print(f"User clicked {clicked_recipe}，Similar Recipe：")
print(recommended_recipes)

# 计算并输出平均相似度
avg_similarity, _ = compute_avg_similarity(clicked_recipe, similarity_matrix, df, similar_recipes_faiss)
print(f"Average Similarity: {avg_similarity:.4f}")

# 绘制相似度分布
plot_similarity_distribution(clicked_recipe, similarity_matrix, df, similar_recipes_faiss)
