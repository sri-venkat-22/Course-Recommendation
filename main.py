"""coursera-course-recommendation-system-webapp.ipynb"""

# Import Dependencies
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import pickle

# # Ensure NLTK resources are downloaded
nltk.download('punkt_tab')
# nltk.download('stopwords')

print('Packages Imported')

# Loading the Data
try:
    data = pd.read_csv("Coursera.csv")
except FileNotFoundError:
    print("Error: Coursera.csv file not found. Please ensure the file is in the correct directory.")
    exit()

# Selecting relevant columns for processing
data = data[['Course Name', 'Difficulty Level', 'Course Description', 'Skills']]

# Data Preprocessing
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Removing special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalizing whitespace
    return text

# Cleaning all relevant columns
for column in data.columns:
    data[column] = data[column].apply(clean_text)

# Creating tags for better combination logic
data['tags'] = (data['Course Name'] + ' ' +
                data['Difficulty Level'] + ' ' +
                data['Course Description'] + ' ' +
                data['Skills'])

# Final dataframe
new_df = data[['Course Name', 'tags']].copy()
new_df.columns = ['course_name', 'tags']

# Improved Stemming Process
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def process_text(text):
    # Tokenizing and stemming
    words = nltk.word_tokenize(text)
    # Filtering stopwords and stem
    stemmed = [ps.stem(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(stemmed)

new_df['tags'] = new_df['tags'].apply(process_text)


tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    min_df=2,
    max_df=0.95
)

try:
    vectors = tfidf.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
except ValueError as e:
    print(f"Error in vectorization: {e}")
    exit()

# Recommendation Function
def recommend(course, num_recommendations=6):
    try:
        # Find course index
        course_matches = new_df[new_df['course_name'].str.contains(course, case=False)]
        if len(course_matches) == 0:
            return "Course not found. Please check the course name and try again."

        course_index = course_matches.index[0]

        distances = similarity[course_index]
        # Sorting and getting top recommendations
        course_indices = sorted(
            [(i, score) for i, score in enumerate(distances) if i != course_index],
            key=lambda x: x[1],
            reverse=True
        )[:num_recommendations]

        # Returning recommendations
        recommendations = []
        for idx, score in course_indices:
            rec = {
                'course_name': new_df.iloc[idx]['course_name'],
                'similarity_score': round(score * 100, 2)
            }
            recommendations.append(rec)

        return recommendations

    except Exception as e:
        return f"Error in recommendation: {str(e)}"

if __name__ == "__main__":
    sample_course = "business strategy business model canvas analysis with miro"
    results = recommend(sample_course)

    if isinstance(results, list):
        print(f"Recommendations for '{sample_course}':")
        for rec in results:
            print(f"- {rec['course_name']} (Similarity: {rec['similarity_score']}%)")
    else:
        print(results)

    # Export model
    try:
        pickle.dump({
            'similarity': similarity,
            'courses': new_df.to_dict(),
            'vectorizer': tfidf
        }, open('course_recommender_model.pkl', 'wb'))
        print("Model exported successfully")
    except Exception as e:
        print(f"Error exporting model: {e}")
