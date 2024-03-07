

import numpy as np 
import pandas as pd 



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



"""# **LOAD & CHECK DATA**"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import re
from PIL import Image
import requests
import random
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

books=pd.read_csv("Books.csv")
books.head(3)

ratings=pd.read_csv("Ratings.csv")
ratings.head(3)

users=pd.read_csv("Users.csv")
users.head(3)

print("Books Shape: " ,books.shape )
print("Ratings Shape: " ,ratings.shape )
print("Users Shape: " ,users.shape )

print("Any null values in Books:\n" ,books.isnull().sum())
print("Any null values in Ratings:\n ",ratings.isnull().sum())
print("Any null values in Users:\n",users.isnull().sum())

"""# **PREPROCESSING**"""

books_data=books.merge(ratings,on="ISBN")
books_data.head()

df=books_data.copy()
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df.drop(columns=["ISBN","Year-Of-Publication","Image-URL-S","Image-URL-M"],axis=1,inplace=True)
df.drop(index=df[df["Book-Rating"]==0].index,inplace=True)
df["Book-Title"]=df["Book-Title"].apply(lambda x: re.sub("[\W_]+"," ",x).strip())
df.head()

"""# **POPULARITY BASED RECOMMENDATION SYSTEM**

* Popularity based recommendation systems are based on the rating of items by all the users.
* Popularity based recommendation systems works with the trend. It basically uses the items which are in trend right now.
"""

def popular_books(df,n=100):
    rating_count=df.groupby("Book-Title").count()["Book-Rating"].reset_index()
    rating_count.rename(columns={"Book-Rating":"NumberOfVotes"},inplace=True)

    rating_average=df.groupby("Book-Title")["Book-Rating"].mean().reset_index()
    rating_average.rename(columns={"Book-Rating":"AverageRatings"},inplace=True)

    popularBooks=rating_count.merge(rating_average,on="Book-Title")

    def weighted_rate(x):
        v=x["NumberOfVotes"]
        R=x["AverageRatings"]

        return ((v*R) + (m*C)) / (v+m)

    C=popularBooks["AverageRatings"].mean()
    m=popularBooks["NumberOfVotes"].quantile(0.90)

    popularBooks=popularBooks[popularBooks["NumberOfVotes"] >=250]
    popularBooks["Popularity"]=popularBooks.apply(weighted_rate,axis=1)
    popularBooks=popularBooks.sort_values(by="Popularity",ascending=False)
    return popularBooks[["Book-Title","NumberOfVotes","AverageRatings","Popularity"]].reset_index(drop=True).head(n)

import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt



n = 10
top_ten = pd.DataFrame(popular_books(df, n))
fig, ax = plt.subplots(1, n, figsize=(17, 5))
fig.suptitle("MOST POPULAR 10 BOOKS", fontsize=40, color="deepskyblue")

# Kullanıcı ajanı ayarları
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

for i in range(len(top_ten["Book-Title"].tolist())):
    url = df.loc[df["Book-Title"] == top_ten["Book-Title"].tolist()[i], "Image-URL-L"][:1].values[0]
    response = requests.get(url, headers=headers, stream=True)

    # İstek başarılı mı diye kontrol et
    if response.status_code == 200:
        # Resmi BytesIO nesnesine yaz ve Image.open ile aç
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        ax[i].imshow(img)
        ax[i].axis("off")
        ax[i].set_title(f"RATING: {round(df[df['Book-Title'] == top_ten['Book-Title'].tolist()[i]]['Book-Rating'].mean(), 1)}", y=-0.20, color="mediumorchid", fontsize=10)
    else:
        print(f"Resim yüklenemedi: Durum Kodu {response.status_code}")
        

plt.show()

"""# **ITEM-BASED COLLABORATIVE FILTERING**"""

import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt

def item_based(bookTitle, df):
    bookTitle = str(bookTitle)

    if bookTitle in df["Book-Title"].values:
        rating_count = pd.DataFrame(df["Book-Title"].value_counts())
        rare_books = rating_count[rating_count["Book-Title"] <= 200].index
        common_books = df[~df["Book-Title"].isin(rare_books)]

        if bookTitle in rare_books:
            most_common = pd.Series(common_books["Book-Title"].unique()).sample(3).values
            print("No Recommendations for this Book ☹️ \n")
            print("YOU MAY TRY: \n")
            for book in most_common:
                print(book, "\n")
        else:
            common_books_pivot = common_books.pivot_table(index=["User-ID"], columns=["Book-Title"], values="Book-Rating")
            title = common_books_pivot[bookTitle]
            recommendation_df = pd.DataFrame(common_books_pivot.corrwith(title).sort_values(ascending=False)).reset_index(drop=False)

            if bookTitle in [title for title in recommendation_df["Book-Title"]]:
                recommendation_df = recommendation_df.drop(recommendation_df[recommendation_df["Book-Title"] == bookTitle].index[0])

            less_rating = []
            for i in recommendation_df["Book-Title"]:
                if df[df["Book-Title"] == i]["Book-Rating"].mean() < 5:
                    less_rating.append(i)
            if recommendation_df.shape[0] - len(less_rating) > 5:
                recommendation_df = recommendation_df[~recommendation_df["Book-Title"].isin(less_rating)]

            recommendation_df = recommendation_df[0:5]
            recommendation_df.columns = ["Book-Title", "Correlation"]

            fig, ax = plt.subplots(1, 5, figsize=(17, 5))
            fig.suptitle("WOULD YOU LIKE to TRY THESE BOOKS?", fontsize=40, color="deepskyblue")

            # Kullanıcı ajanı ayarları
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }

            for i in range(len(recommendation_df["Book-Title"].tolist())):
                url = df.loc[df["Book-Title"] == recommendation_df["Book-Title"].tolist()[i], "Image-URL-L"][:1].values[0]
                response = requests.get(url, headers=headers, stream=True)

                if response.status_code == 200:
                    img_data = BytesIO(response.content)
                    img = Image.open(img_data)
                    ax[i].imshow(img)
                    ax[i].axis("off")
                    ax[i].set_title(f"RATING: {round(df[df['Book-Title'] == recommendation_df['Book-Title'].tolist()[i]]['Book-Rating'].mean(), 1)}", y=-0.20, color="mediumorchid", fontsize=22)
                else:
                    print(f"Resim yüklenemedi: Durum Kodu {response.status_code}")
                    

            plt.show()
    else:
        print("❌ COULD NOT FIND ❌")



item_based("Me Talk Pretty One Day",df)

item_based("From One to One Hundred",df)

item_based("The Da Vinci Code",df)

item_based("Barbie",df)

"""# **USER-BASED COLLABORATIVE FILTERING**"""

new_df=df[df['User-ID'].map(df['User-ID'].value_counts()) > 200]  # Drop users who vote less than 200 times.
users_pivot=new_df.pivot_table(index=["User-ID"],columns=["Book-Title"],values="Book-Rating")
users_pivot.fillna(0,inplace=True)

def users_choice(id):

    users_fav=new_df[new_df["User-ID"]==id].sort_values(["Book-Rating"],ascending=False)[0:5]
    return users_fav

def user_based(new_df,id):
    if id not in new_df["User-ID"].values:
        print("❌ User NOT FOUND ❌")


    else:
        index=np.where(users_pivot.index==id)[0][0]
        similarity=cosine_similarity(users_pivot)
        similar_users=list(enumerate(similarity[index]))
        similar_users = sorted(similar_users,key = lambda x:x[1],reverse=True)[0:5]

        user_rec=[]

        for i in similar_users:
                data=df[df["User-ID"]==users_pivot.index[i[0]]]
                user_rec.extend(list(data.drop_duplicates("User-ID")["User-ID"].values))

    return user_rec

def common(new_df,user,user_id):
    x=new_df[new_df["User-ID"]==user_id]
    recommend_books=[]
    user=list(user)
    for i in user:
        y=new_df[(new_df["User-ID"]==i)]
        books=y.loc[~y["Book-Title"].isin(x["Book-Title"]),:]
        books=books.sort_values(["Book-Rating"],ascending=False)[0:5]
        recommend_books.extend(books["Book-Title"].values)

    return recommend_books[0:5]

import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import random

# Kullanıcı ajanı ayarları
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Favori kitaplarını gösteren bölüm
user_id = random.choice(new_df["User-ID"].values)
user_choice_df = pd.DataFrame(users_choice(user_id))
user_favorite = users_choice(user_id)
n = len(user_choice_df["Book-Title"].values)
print("🟦 USER: {} ".format(user_id))

fig, ax = plt.subplots(1, n, figsize=(17, 5))
fig.suptitle("YOUR FAVORITE BOOKS", fontsize=40, color="salmon")

for i in range(n):
    url = new_df.loc[new_df["Book-Title"] == user_choice_df["Book-Title"].tolist()[i], "Image-URL-L"][:1].values[0]
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        ax[i].imshow(img)
        ax[i].axis("off")
        ax[i].set_title("RATING: {} ".format(round(new_df[new_df["Book-Title"] == user_choice_df["Book-Title"].tolist()[i]]["Book-Rating"].mean(), 1)), y=-0.20, color="mediumorchid", fontsize=22)
    else:
        print(f"Resim yüklenemedi: Durum Kodu {response.status_code}")

plt.show()

# Önerilen kitaplarını gösteren bölüm
user_based_rec = user_based(new_df, user_id)
books_for_user = common(new_df, user_based_rec, user_id)
books_for_userDF = pd.DataFrame(books_for_user, columns=["Book-Title"])

fig, ax = plt.subplots(1, 5, figsize=(17, 5))
fig.suptitle("YOU MAY ALSO LIKE THESE BOOKS", fontsize=40, color="mediumseagreen")

for i in range(5):
    url = new_df.loc[new_df["Book-Title"] == books_for_userDF["Book-Title"].tolist()[i], "Image-URL-L"][:1].values[0]
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        ax[i].imshow(img)
        ax[i].axis("off")
        ax[i].set_title("RATING: {} ".format(round(new_df[new_df["Book-Title"] == books_for_userDF["Book-Title"].tolist()[i]]["Book-Rating"].mean(), 1)), y=-0.20, color="mediumorchid", fontsize=22)
    else:
        print(f"Resim yüklenemedi: Durum Kodu {response.status_code}")

plt.show()

"""# **CONTENT-BASED COLLABORATIVE FILTERING**"""

import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based(bookTitle, df):
    bookTitle = str(bookTitle)

    if bookTitle in df["Book-Title"].values:
        rating_count = pd.DataFrame(df["Book-Title"].value_counts())
        rare_books = rating_count[rating_count["Book-Title"] <= 200].index
        common_books = df[~df["Book-Title"].isin(rare_books)]

        if bookTitle in rare_books:
            most_common = pd.Series(common_books["Book-Title"].unique()).sample(3).values
            print("No Recommendations for this Book ☹️ \n")
            print("YOU MAY TRY: \n")
            for book in most_common:
                print(book, "\n")
        else:
            common_books = common_books.drop_duplicates(subset=["Book-Title"])
            common_books.reset_index(inplace=True)
            common_books["index"] = [i for i in range(common_books.shape[0])]
            targets = ["Book-Title", "Book-Author", "Publisher"]
            common_books["all_features"] = [" ".join(common_books[targets].iloc[i,].values) for i in range(common_books[targets].shape[0])]
            vectorizer = CountVectorizer()
            common_booksVector = vectorizer.fit_transform(common_books["all_features"])
            similarity = cosine_similarity(common_booksVector)
            index = common_books[common_books["Book-Title"] == bookTitle]["index"].values[0]
            similar_books = list(enumerate(similarity[index]))
            similar_booksSorted = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:6]
            books = []
            for i in range(len(similar_booksSorted)):
                books.append(common_books[common_books["index"] == similar_booksSorted[i][0]]["Book-Title"].item())
            fig, ax = plt.subplots(1, 5, figsize=(17, 5))
            fig.suptitle("YOU MAY ALSO LIKE THESE BOOKS", fontsize=40, color="chocolate")

            # Kullanıcı ajanı ayarları
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }

            for i in range(len(books)):
                url = common_books.loc[common_books["Book-Title"] == books[i], "Image-URL-L"][:1].values[0]
                response = requests.get(url, headers=headers, stream=True)
                if response.status_code == 200:
                    img_data = BytesIO(response.content)
                    img = Image.open(img_data)
                    ax[i].imshow(img)
                    ax[i].axis("off")
                    ax[i].set_title("RATING: {}".format(round(df[df["Book-Title"] == books[i]]["Book-Rating"].mean(), 1)), y=-0.20, color="mediumorchid", fontsize=22)
                else:
                    print(f"Resim yüklenemedi: Durum Kodu {response.status_code}")
            plt.show()

    else:
        print("❌ COULD NOT FIND ❌")



content_based("The Da Vinci Code",df)

content_based("Tuesdays with Morrie An Old Man a Young Man and Life s Greatest Lesson",df)

content_based("A Soldier of the Great War",df)

content_based("Life of Pi",df)