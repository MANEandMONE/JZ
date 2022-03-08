from django.shortcuts import render
import json

from django.http import HttpResponse, request, response
from django.contrib.auth.hashers import make_password
# from .models import User
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import sklearn as sk
import warnings
# 직렬화
from rest_framework import viewsets
import csv
import random
from .models import *
# def register(request):   #회원가입 페이지를 보여주기 위한 함수
#     if request.method == "GET" :
#         return render(request, 'beer/register.html') #register를 요청받으면 register.html 로 응답

#     elif request.method == "POST":
#         username = request.POST.get('username',None)   #딕셔너리형태
#         password = request.POST.get('password',None)
#         re_password = request.POST.get('re_password',None)
#         res_data = {}
#         if not (username and password and re_password) :
#             res_data['error'] = "모든 값을 입력해야 합니다."
#         if password != re_password :
#             # return HttpResponse('비밀번호가 다릅니다.')
#             res_data['error'] = '비밀번호가 다릅니다.'
#         else :
#             user = User(username=username, password=make_password(password))
#             user.save()
#         return render(request, 'beer/register.html', res_data) #register를 요청받으면 register.html 로 응답.
warnings.filterwarnings('ignore')

# Viewset API Set


# 우리가 예측한 평점과 실제 평점간의 차이를 MSE로 계산
def get_mse(pred, actual):
    # 평점이 있는 실제 영화만 추출
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# 특정 맥주와 비슷한 유사도를 가지는 맥주 Top_N에 대해서만 적용 -> 시간오래걸림
def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
    # 사용자-아이템 평점 행렬 크기만큼 0으로 채운 예측 행렬 초기화
    pred = np.zeros(ratings_arr.shape)

    # 사용자-아이템 평점 행렬의 맥주 개수만큼 루프
    for col in range(ratings_arr.shape[1]):
        # 유사도 행렬에서 유사도가 큰 순으로 n개의 데이터 행렬의 인덱스 반환
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n - 1:-1]]
        # 개인화된 예측 평점 계산 : 각 col 맥주별(1개), 2496 사용자들의 예측평점
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(
                ratings_arr[row, :][top_n_items].T)
            pred[row, col] /= np.sum(item_sim_arr[col, :][top_n_items])

    return pred


# 사용자가 안 먹어본 맥주를 추천하자.


def get_not_tried_beer(ratings_matrix, userId):
    # userId로 입력받은 사용자의 모든 맥주 정보를 추출해 Series로 반환
    # 반환된 user_rating은 영화명(title)을 인덱스로 가지는 Series 객체
    user_rating = ratings_matrix.loc[userId, :]

    # user_rating이 0보다 크면 기존에 관란함 영화.
    # 대상 인덱스를 추출해 list 객체로 만듦
    tried = user_rating[user_rating > 0].index.tolist()

    # 모든 맥주명을 list 객체로 만듦
    beer_list = ratings_matrix.columns.tolist()

    # list comprehension으로 tried에 해당하는 영화는 beer_list에서 제외
    not_tried = [beer for beer in beer_list if beer not in tried]

    return not_tried


# 예측 평점 DataFrame에서 사용자 id 인덱스와 not_tried로 들어온 맥주명 추출 후
# 가장 예측 평점이 높은 순으로 정렬


def recomm_beer_by_userid(pred_df, userId, not_tried, top_n):
    recomm_beer = pred_df.loc[userId,
                              not_tried].sort_values(ascending=False)[:top_n]
    return recomm_beer


# 평점, Aroma, Flavor, Mouthfeel 중 피처 선택 후 유사도 계산


def recomm_feature(df):

    ratings = df[['장소', '아이디', '평점']]
    # 피벗 테이블을 이용해 유저-아이디 매트릭스 구성
    ratings_matrix = ratings.pivot_table('평점', index='아이디', columns='장소')
    ratings_matrix.head(3)

    # fillna함수를 이용해 Nan처리
    ratings_matrix = ratings_matrix.fillna(0)

    # 유사도 계산을 위해 트랜스포즈
    ratings_matrix_T = ratings_matrix.transpose()

    # 아이템-유저 매트릭스로부터 코사인 유사도 구하기
    item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)

    # cosine_similarity()로 반환된 넘파이 행렬에 영화명을 매핑해 DataFrame으로 변환
    item_sim_df = pd.DataFrame(data=item_sim,
                               index=ratings_matrix.columns,
                               columns=ratings_matrix.columns)

    return item_sim_df


# 해당 맥주와 유사한 유사도 5개 추천


def recomm_beer(item_sim_df, beer_name):
    # 해당 맥주와 유사도가 높은 맥주 5개만 추천
    return item_sim_df[beer_name].sort_values(ascending=False)[0:10]


def recomm_detail(item_sim_df, detail):
    # 해당 맥주와 유사도가 높은 맥주 5개만 추천
    return item_sim_df[detail].sort_values(ascending=False)[0:10]


def index(request):
    return render(request, 'beer/index.html')


def ver1(request):
    beer_list = pd.read_csv('도시이름.csv', encoding='utf-8', index_col=0)
    ratings = pd.read_csv('총 평점.csv', encoding='utf-8', index_col=0)
    cluster_3 = pd.read_csv('대표군집클러스터링.csv', encoding='utf-8', index_col=0)
    cluster_all = pd.read_csv('전체도시클러스터링.csv', encoding='utf-8', index_col=0)
    beer_list = beer_list['city']

    if request.method == 'POST':
        beer_name = request.POST.get('beer', '')

        df = recomm_feature(ratings)

        result = recomm_beer(df, beer_name)
        result = result.index.tolist()
        random.shuffle(result)
        return render(request, 'beer/ver1_result.html', {
            'result': result,
            'beer_list': beer_list,
        })
    else:
        return render(request, 'beer/ver1.html', {'beer_list': beer_list})


# def vr0(requset):
#     beer_list = pd.read_csv('도시이름.csv', encoding='utf-8', index_col=0)
#     beer_list = beer_list['city']
#     return render(request, 'beer/vr0.html')

# def vr0(response):
#     return render(response, 'beer/vr0.html')


def ver3(request):
    df_cluster = pd.read_csv('result.csv', encoding='utf-8', index_col=0)

    cst0_list = df_cluster.loc[df_cluster['Cluster'] == 0, 'locate'].tolist()

    cst1_list = df_cluster.loc[df_cluster['Cluster'] == 1, 'locate'].tolist()

    cst2_list = df_cluster.loc[df_cluster['Cluster'] == 2, 'locate'].tolist()

    cst3_list = df_cluster.loc[df_cluster['Cluster'] == 3, 'locate'].tolist()

    cst4_list = df_cluster.loc[df_cluster['Cluster'] == 4, 'locate'].tolist()

    cst5_list = df_cluster.loc[df_cluster['Cluster'] == 5, 'locate'].tolist()

    cst6_list = df_cluster.loc[df_cluster['Cluster'] == 6, 'locate'].tolist()

    cst7_list = df_cluster.loc[df_cluster['Cluster'] == 7, 'locate'].tolist()

    cst8_list = df_cluster.loc[df_cluster['Cluster'] == 8, 'locate'].tolist()

    cst9_list = df_cluster.loc[df_cluster['Cluster'] == 9, 'locate'].tolist()

    cst10_list = df_cluster.loc[df_cluster['Cluster'] == 10, 'locate'].tolist()

    cst11_list = df_cluster.loc[df_cluster['Cluster'] == 11, 'locate'].tolist()

    if request.method == 'POST':
        detail = request.POST.get('detail', '')

        if detail in ['food', 'walk', 'nature']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'culture']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'date']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'sleep']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'drive']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'night']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'culture']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'date']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'sleep']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'drive']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'night']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'date']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'sleep']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'drive']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'night']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'family']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'view']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'sleep']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'drive']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'night']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['food', 'sleep', 'drive']:
            result = cst10_list
            random.shuffle(result)

        elif detail in ['food', 'sleep', 'night']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['food', 'sleep', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'sleep', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['food', 'sleep', 'family']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'sleep', 'view']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'drive', 'night']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'drive', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'drive', 'sns']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'drive', 'family']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['food', 'drive', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'night', 'fori']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['food', 'night', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'night', 'family']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['food', 'night', 'view']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['food', 'fori', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'fori', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'sns', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'sns', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['food', 'family', 'view']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'culture']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'date']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'sleep']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'drive']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'night']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'fori']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'view']:
            result = cst_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'date']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'sleep']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'drive']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'night']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'family']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'sleep']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'drive']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'night']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'sns']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['walk', 'sleep', 'drive']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['walk', 'sleep', 'night']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'sleep', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'sleep', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'sleep', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'sleep', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['walk', 'drive', 'night']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['walk', 'drive', 'fori']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['walk', 'drive', 'sns']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'drive', 'family']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['walk', 'drive', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'night', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'night', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['walk', 'night', 'family']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'night', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'fori', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['walk', 'fori', 'family']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['walk', 'fori', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'sns', 'family']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['walk', 'sns', 'view']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['walk', 'family', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'date']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'sleep']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'drive']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'night']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'sns']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'family']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'view']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'sleep']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'drive']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'night']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['nature', 'sleep', 'drive']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'sleep', 'night']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['nature', 'sleep', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'sleep', 'sns']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['nature', 'sleep', 'family']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'sleep', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['nature', 'drive', 'night']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['nature', 'drive', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'drive', 'sns']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'drive', 'family']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'drive', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['nature', 'night', 'fori']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['nature', 'night', 'sns']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['nature', 'night', 'family']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'night', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'fori', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'fori', 'family']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'fori', 'view']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['nature', 'sns', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['nature', 'sns', 'view']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'family', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'sleep']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'drive']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'night']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'family']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'view']:
            result = cst10_list
            random.shuffle(result)

        elif detail in ['culture', 'sleep', 'drive']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['culture', 'sleep', 'night']:
            result = cst_list
            random.shuffle(result)

        elif detail in ['culture', 'sleep', 'fori']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['culture', 'sleep', 'sns']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'sleep', 'family']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['culture', 'sleep', 'view']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'drive', 'night']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['culture', 'drive', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['culture', 'drive', 'sns']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['culture', 'drive', 'family']:
            result = cst10_list
            random.shuffle(result)

        elif detail in ['culture', 'drive', 'view']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['culture', 'night', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['culture', 'night', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['culture', 'night', 'family']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['culture', 'night', 'view']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'fori', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['culture', 'fori', 'family']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['culture', 'fori', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['culture', 'sns', 'family']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'sns', 'view']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['culture', 'family', 'view']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['date', 'sleep', 'drive']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['date', 'sleep', 'night']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['date', 'sleep', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['date', 'sleep', 'sns']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['date', 'sleep', 'family']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['date', 'sleep', 'view']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['date', 'drive', 'night']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['date', 'drive', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['date', 'drive', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['date', 'drive', 'family']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['date', 'drive', 'view']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['date', 'night', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['date', 'night', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['date', 'night', 'family']:
            result = cst10_list
            random.shuffle(result)

        elif detail in ['date', 'night', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['date', 'fori', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['date', 'fori', 'family']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['date', 'fori', 'view']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['date', 'sns', 'family']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['date', 'sns', 'view']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['date', 'family', 'view']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['sleep', 'drive', 'night']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['sleep', 'drive', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['sleep', 'drive', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['sleep', 'drive', 'family']:
            result = cst10_list
            random.shuffle(result)

        elif detail in ['sleep', 'drive', 'view']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['sleep', 'night', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['sleep', 'night', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['sleep', 'night', 'family']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['sleep', 'night', 'view']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['sleep', 'fori', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['sleep', 'fori', 'family']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['sleep', 'fori', 'view']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['sleep', 'sns', 'family']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['sleep', 'sns', 'view']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['sleep', 'family', 'view']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['drive', 'night', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['drive', 'night', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['drive', 'night', 'family']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['drive', 'night', 'view']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['drive', 'sns', 'family']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['drive', 'sns', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['drive', 'family', 'view']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['night', 'fori', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['night', 'fori', 'family']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['night', 'fori', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['night', 'sns', 'family']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['night', 'sns', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['night', 'family', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['fori', 'sns', 'family']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['fori', 'sns', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['fori', 'family', 'view']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['sns', 'family', 'view']:
            result = cst2_list
            random.shuffle(result)

        return render(request, 'beer/ver3_result.html', {
            'result': result,
        })
    else:
        return render(request, 'beer/ver3.html')


def hotel(request):
    result = ['양재시민의숲', '경복궁', '쇠소깍', '천제연폭포']
    datas1 = Hotel.objects.filter(locate=result[0])
    datas2 = Hotel.objects.filter(locate=result[1])
    datas3 = Hotel.objects.filter(locate=result[2])
    datas4 = Hotel.objects.filter(locate=result[3])

    datas5 = Restaurant.objects.filter(locate=result[0])
    datas6 = Restaurant.objects.filter(locate=result[1])
    datas7 = Restaurant.objects.filter(locate=result[2])
    datas8 = Restaurant.objects.filter(locate=result[3])

    return render(
        request, 'beer/hotel.html', {
            'hotels1': datas1,
            'hotels2': datas2,
            'hotels3': datas3,
            'hotels4': datas4,
            'restaurant1': datas5,
            'restaurant2': datas6,
            'restaurant3': datas7,
            'restaurant4': datas8,
        })
