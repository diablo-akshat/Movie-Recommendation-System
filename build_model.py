import os
import ast
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_field(x):
    """Parse a stringified list of dicts and return list of names."""
    if pd.isna(x):
        return []
    try:
        data = ast.literal_eval(x)
    except Exception:
        try:
            # fallback: sometimes quotes are doubled inside CSV
            data = ast.literal_eval(x.replace('""', '"'))
        except Exception:
            return []
    names = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = item.get('name') or item.get('title')
                if name:
                    names.append(str(name))
    return names


def get_cast(x, top_n=3):
    names = parse_field(x)
    # keep only first `top_n` cast names
    return names[:top_n]


def get_crew_director(x):
    if pd.isna(x):
        return []
    try:
        data = ast.literal_eval(x)
    except Exception:
        try:
            data = ast.literal_eval(x.replace('""', '"'))
        except Exception:
            return []
    directors = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get('job') and item.get('job').lower() == 'director':
                directors.append(item.get('name'))
    return directors


def clean_token_list(lst):
    # make tokens lowercase and remove inner spaces so multi-word names become single tokens
    return [t.replace(' ', '').lower() for t in lst if isinstance(t, str)]


def main():
    movies_csv = 'Movies 500.csv'
    credits_csv = 'Credits 500.csv'

    if not (os.path.exists(movies_csv) and os.path.exists(credits_csv)):
        print('Required CSV files not found. Place "Movies 500.csv" and "Credits 500.csv" here.')
        return

    movies = pd.read_csv(movies_csv)
    credits = pd.read_csv(credits_csv)

    # ensure id field is present
    movies['movie_id'] = movies['id']

    # parse genres and keywords
    movies['genres_parsed'] = movies['genres'].apply(parse_field)
    movies['keywords_parsed'] = movies['keywords'].apply(parse_field)

    # merge credits to get cast and crew
    credits = credits.rename(columns={'movie_id': 'id'})
    merged = movies.merge(credits[['id', 'cast', 'crew']], left_on='id', right_on='id', how='left')

    merged['cast_parsed'] = merged['cast'].apply(get_cast)
    merged['director_parsed'] = merged['crew'].apply(get_crew_director)

    # build tags: overview + genres + keywords + cast + director
    def make_tags(row):
        overview = row.get('overview') or ''
        genres = clean_token_list(row.get('genres_parsed') or [])
        keywords = clean_token_list(row.get('keywords_parsed') or [])
        cast = clean_token_list(row.get('cast_parsed') or [])
        director = clean_token_list(row.get('director_parsed') or [])
        parts = [overview] + genres + keywords + cast + director
        return ' '.join([str(p) for p in parts if p])

    merged['tags'] = merged.apply(make_tags, axis=1)

    # vectorize
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(merged['tags'].astype('str'))

    # similarity
    similarity = cosine_similarity(vectors)

    # prepare model dir
    os.makedirs('model', exist_ok=True)

    # ensure DataFrame has `title` and `movie_id` columns as expected by app.py
    out_movies = merged.copy()
    # save the dataframe and similarity
    with open(os.path.join('model', 'movie_list.pkl'), 'wb') as f:
        pickle.dump(out_movies, f)

    with open(os.path.join('model', 'similarity.pkl'), 'wb') as f:
        pickle.dump(similarity, f)

    print('Built model files: model/movie_list.pkl and model/similarity.pkl')


if __name__ == '__main__':
    main()
