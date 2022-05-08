from django.shortcuts import render, redirect
from django.db import transaction
from django.core.paginator import Paginator

from django.core.serializers.json import DjangoJSONEncoder

from django.http import HttpResponse, request, response

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

from django.conf import settings
from user.models import User

warnings.filterwarnings('ignore')


# 우리가 예측한 평점과 실제 평점간의 차이를 MSE로 계산
def get_mse(pred, actual):
    # 평점이 있는 실제 영화만 추출
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# 특정 도시와 비슷한 유사도를 가지는 도시 Top_N에 대해서만 적용 -> 시간오래걸림
def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
    # 사용자-아이템 평점 행렬 크기만큼 0으로 채운 예측 행렬 초기화
    pred = np.zeros(ratings_arr.shape)

    # 사용자-아이템 평점 행렬의 도시 개수만큼 루프
    for col in range(ratings_arr.shape[1]):
        # 유사도 행렬에서 유사도가 큰 순으로 n개의 데이터 행렬의 인덱스 반환
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n - 1:-1]]
        # 개인화된 예측 평점 계산 : 각 col 도시별(1개), 2496 사용자들의 예측평점
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(
                ratings_arr[row, :][top_n_items].T)
            pred[row, col] /= np.sum(item_sim_arr[col, :][top_n_items])

    return pred


def get_not_tried_beer(ratings_matrix, userId):
    # userId로 입력받은 사용자의 모든 도시 정보를 추출해 Series로 반환
    # 반환된 user_rating은 영화명(title)을 인덱스로 가지는 Series 객체
    user_rating = ratings_matrix.loc[userId, :]

    # user_rating이 0보다 크면 기존에 관란함 영화.
    # 대상 인덱스를 추출해 list 객체로 만듦
    tried = user_rating[user_rating > 0].index.tolist()

    # 모든 도시명을 list 객체로 만듦
    beer_list = ratings_matrix.columns.tolist()

    # list comprehension으로 tried에 해당하는 도시는 beer_list에서 제외
    not_tried = [beer for beer in beer_list if beer not in tried]

    return not_tried


# 예측 평점 DataFrame에서 사용자 id 인덱스와 not_tried로 들어온 도시명 추출 후
# 가장 예측 평점이 높은 순으로 정렬


def recomm_beer_by_userid(pred_df, userId, not_tried, top_n):
    recomm_beer = pred_df.loc[userId,
                              not_tried].sort_values(ascending=False)[:top_n]
    return recomm_beer


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


def recomm_beer(item_sim_df, beer_name):
    # 해당 도시와 유사도가 높은 도시 5개만 추천
    return item_sim_df[beer_name].sort_values(ascending=False)[1:10]


def recomm_detail(item_sim_df, detail):
    # 해당 도시와 유사도가 높은 도시 5개만 추천
    return item_sim_df[detail].sort_values(ascending=False)[1:10]


# 선택한 관광지 세션 저장
def ver2(request):
    beer_list = pd.read_csv('result.csv', encoding='utf-8', index_col=0)

    beer_list = beer_list['locate']

    text = {'beer_list': beer_list}

    login_session = request.session.get('login_session')

    if login_session == '':
        text['login_session'] = False
    else:
        text['login_session'] = True
    if request.method == 'POST':
        beer_name = request.POST.get('beer', '')
        request.session['tour'] = beer_name
        text['tour'] = request.session['tour']

    return render(request, 'beer/ver2.html', text)


# 세션에 저장된 관광지 가져와 계산
def ver2_session(request):
    ratings = pd.read_csv('merge.csv', encoding='utf-8', index_col=0)
    cluster_3 = pd.read_csv('대표군집클러스터링.csv', encoding='utf-8', index_col=0)
    cluster_all = pd.read_csv('전체도시클러스터링.csv', encoding='utf-8', index_col=0)

    # 세션 데이터 가져오기
    beer_name = request.session.get('tour')

    # 계산
    df = recomm_feature(ratings)

    result = recomm_beer(df, beer_name)
    result = result.index.tolist()

    # 로그인 세션 유지
    login_session = request.session.get('login_session')

    if login_session == '':
        request.session['login_session'] = False
    else:
        request.session['login_session'] = True

    # 숙박 시설 필터
    # 호텔 모텔 펜션 리조트 게스트하우스 호스텔
    cost = request.GET.get('cost', '')
    sort = request.GET.get('sort', '')
    rating = request.GET.get('rating', '')
    distance = request.GET.get('distance', '')
    review = request.GET.get('review', '')

    if rating == 'rating':
        content_list = Hotel.objects.filter(
            place=result[0]).order_by('-rating')

    elif distance == 'distance':
        content_list = hotel1_distance_up = Hotel.objects.filter(
            place=result[0]).order_by('distance')

    elif cost == 'cost_down':
        content_list = Hotel.objects.filter(place=result[0]).order_by('-cost')

    elif cost == 'cost_up':
        content_list = Hotel.objects.filter(place=result[0]).order_by('cost')

    elif sort == 'hotell':
        content_list = Hotel.objects.filter(place=result[0],
                                            classfication='호텔')

    elif sort == 'hotell' and cost == 'cost_down':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='호텔').order_by('-cost')

    elif sort == 'hotell' and cost == 'cost_up':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='호텔').order_by('cost')

    elif sort == 'hotell' and rating == 'rating':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='호텔').order_by('-rating')

    elif sort == 'hotell' and distance == 'distance':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='호텔').order_by('distance')

    elif sort == 'guesthouse':
        content_list = Hotel.objects.filter(place=result[0],
                                            classfication='호텔')

    elif sort == 'guesthouse' and cost == 'cost_down':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='게스트하우스').order_by('-cost')

    elif sort == 'guesthouse' and cost == 'cost_up':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='게스트하우스').order_by('cost')

    elif sort == 'guesthouse' and rating == 'rating':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='게스트하우스').order_by('-rating')

    elif sort == 'guesthouse' and distance == 'distance':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='게스트하우스').order_by('distance')

    elif sort == 'hostel':
        content_list = Hotel.objects.filter(place=result[0],
                                            classfication='호스텔')

    elif sort == 'hostel' and cost == 'cost_down':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='호스텔').order_by('-cost')

    elif sort == 'hostel' and cost == 'cost_up':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='호스텔').order_by('cost')

    elif sort == 'hostel' and rating == 'rating':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='호스텔').order_by('-rating')

    elif sort == 'hostel' and distance == 'distance':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='호스텔').order_by('distance')

    elif sort == 'pension':
        content_list = Hotel.objects.filter(place=result[0],
                                            classfication='펜션')

    elif sort == 'pension' and cost == 'cost_down':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='펜션').order_by('-cost')

    elif sort == 'pension' and cost == 'cost_up':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='펜션').order_by('cost')

    elif sort == 'pension' and rating == 'rating':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='펜션').order_by('-rating')

    elif sort == 'pension' and distance == 'distance':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='펜션').order_by('distance')

    elif sort == 'motel':
        content_list = Hotel.objects.filter(place=result[0],
                                            classfication='모텔')

    elif sort == 'motel' and cost == 'cost_down':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='모텔').order_by('-cost')

    elif sort == 'motel' and cost == 'cost_up':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='모텔').order_by('cost')

    elif sort == 'motel' and rating == 'rating':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='모텔').order_by('-rating')

    elif sort == 'motel' and distance == 'distance':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='모텔').order_by('distance')

    elif sort == 'resort':
        content_list = Hotel.objects.filter(place=result[0],
                                            classfication='리조트')

    elif sort == 'resort' and cost == 'cost_down':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='리조트').order_by('-cost')

    elif sort == 'resort' and cost == 'cost_up':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='리조트').order_by('cost')

    elif sort == 'resort' and rating == 'rating':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='리조트').order_by('-rating')

    elif sort == 'resort' and distance == 'distance':
        content_list = Hotel.objects.filter(
            place=result[0], classfication='리조트').order_by('distance')

    else:
        content_list = Hotel.objects.filter(place=result[0])

    if rating == 'rating':
        content_list1 = Restaurant.objects.filter(
            place=result[0]).order_by('-rating')

    elif review == 'review':
        content_list1 = Restaurant.objects.filter(
            place=result[0]).order_by('-review')
    else:
        content_list1 = Restaurant.objects.filter(place=result[0])

    # 숙박시설 정보 Pagination
    page = request.GET.get('page', 1)
    paginator = Paginator(content_list, 10)
    posts = paginator.get_page(page)

    # 식당정보 필터

    # 식당 정보 Pagination
    page1 = request.GET.get('page', 1)
    paginator1 = Paginator(content_list1, 10)
    posts1 = paginator1.get_page(page1)

    return render(
        request, 'beer/ver2_result.html', {
            'login_session': login_session,
            'result': result,
            'sort': sort,
            'cost': cost,
            'rating': rating,
            'distance': distance,
            'posts': posts,
            'posts1': posts1,
            'content_list': content_list,
            'content_list1': content_list1,
        })


def ver3(request):
    df_cluster = pd.read_csv('result.csv', encoding='utf-8', index_col=0)

    cst0_list = df_cluster.loc[df_cluster['Cluster'] == 0, 'place'].tolist()

    cst1_list = df_cluster.loc[df_cluster['Cluster'] == 1, 'place'].tolist()

    cst2_list = df_cluster.loc[df_cluster['Cluster'] == 2, 'place'].tolist()

    cst3_list = df_cluster.loc[df_cluster['Cluster'] == 3, 'place'].tolist()

    cst4_list = df_cluster.loc[df_cluster['Cluster'] == 4, 'place'].tolist()

    cst5_list = df_cluster.loc[df_cluster['Cluster'] == 5, 'place'].tolist()

    cst6_list = df_cluster.loc[df_cluster['Cluster'] == 6, 'place'].tolist()

    cst7_list = df_cluster.loc[df_cluster['Cluster'] == 7, 'place'].tolist()

    cst8_list = df_cluster.loc[df_cluster['Cluster'] == 8, 'place'].tolist()

    cst9_list = df_cluster.loc[df_cluster['Cluster'] == 9, 'place'].tolist()

    cst10_list = df_cluster.loc[df_cluster['Cluster'] == 10, 'place'].tolist()

    cst11_list = df_cluster.loc[df_cluster['Cluster'] == 11, 'place'].tolist()

    # ver3에서 로그인 세션 유지
    context = {}
    login_session = request.session.get('login_session')

    if login_session == '':
        context['login_session'] = False
    else:
        context['login_session'] = True

    if request.method == 'POST':

        # 결과페이지에서 로그인 세션 유지
        login_session = request.session.get('login_session')

        if login_session == '':
            request.session['login_session'] = False
        else:
            request.session['login_session'] = True

        # detail value POST
        detail = request.POST.get('detail', '')
        detail2 = request.POST.get('topic', )
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

        hotel1 = Hotel.objects.filter(place=result[0])
        hotel1_cost_up = hotel1.order_by('cost')
        hotel1_cost_down = hotel1.order_by('-cost')
        hotel1_rating_up = hotel1.order_by('rating')
        hotel1_rating_down = hotel1.order_by('-rating')
        hotel1_distance_up = hotel1.order_by('distance')
        hotel1_kind_up = hotel1.order_by('kind')
        hotel1_clean_up = hotel1.order_by('clean')
        hotel1_conv_up = hotel1.order_by('conv')
        hotel1_hotel = Hotel.objects.filter(place=result[0],
                                            classfication='호텔')
        hotel1_hostel = Hotel.objects.filter(place=result[0],
                                             classfication='호스텔')
        hotel1_guest = Hotel.objects.filter(place=result[0],
                                            classfication='게스트하우스')
        hotel1_apartment = Hotel.objects.filter(place=result[0],
                                                classfication='아파트')
        hotel1_apartmenthotel = Hotel.objects.filter(place=result[0],
                                                     classfication='아파트호텔')
        hotel1_motel = Hotel.objects.filter(place=result[0],
                                            classfication='모텔')
        hotel1_pension = Hotel.objects.filter(place=result[0],
                                              classfication='펜션')
        hotel1_resort = Hotel.objects.filter(place=result[0],
                                             classfication='리조트')
        hotel1_badandbreakfast = Hotel.objects.filter(place=result[0],
                                                      classfication='베드앤브렉퍼스트')
        hotel1_homestay = Hotel.objects.filter(place=result[0],
                                               classfication='홈스테이')
        hotel1_lodge = Hotel.objects.filter(place=result[0],
                                            classfication='롯지')
        hotel1_countryhouse = Hotel.objects.filter(place=result[0],
                                                   classfication='컨트리하우스')
        hotel1_inn = Hotel.objects.filter(place=result[0], classfication='여관')
        hotel1_villa = Hotel.objects.filter(place=result[0],
                                            classfication='빌라')
        hotel1_camping = Hotel.objects.filter(place=result[0],
                                              classfication='캠핑장')

        paginator = Paginator(hotel1, 10)

        posts = paginator.get_page(page)

        hotel2 = Hotel.objects.filter(place=result[1])
        hotel2_cost_up = hotel2.order_by('cost')
        hotel2_cost_down = hotel2.order_by('-cost')
        hotel2_rating_up = hotel2.order_by('rating')
        hotel2_rating_down = hotel2.order_by('-rating')
        hotel2_distance_up = hotel2.order_by('distance')
        hotel2_kind_up = hotel2.order_by('kind')
        hotel2_clean_up = hotel2.order_by('clean')
        hotel2_conv_up = hotel2.order_by('conv')
        hotel2_hotel = Hotel.objects.filter(place=result[1],
                                            classfication='호텔')
        hotel2_hostel = Hotel.objects.filter(place=result[1],
                                             classfication='호스텔')
        hotel2_guest = Hotel.objects.filter(place=result[1],
                                            classfication='게스트하우스')
        hotel2_apartment = Hotel.objects.filter(place=result[1],
                                                classfication='아파트')
        hotel2_apartmenthotel = Hotel.objects.filter(place=result[1],
                                                     classfication='아파트호텔')
        hotel2_motel = Hotel.objects.filter(place=result[1],
                                            classfication='모텔')
        hotel2_pension = Hotel.objects.filter(place=result[1],
                                              classfication='펜션')
        hotel2_resort = Hotel.objects.filter(place=result[1],
                                             classfication='리조트')
        hotel2_badandbreakfast = Hotel.objects.filter(place=result[1],
                                                      classfication='베드앤브렉퍼스트')
        hotel2_homestay = Hotel.objects.filter(place=result[1],
                                               classfication='홈스테이')
        hotel2_lodge = Hotel.objects.filter(place=result[1],
                                            classfication='롯지')
        hotel2_countryhouse = Hotel.objects.filter(place=result[1],
                                                   classfication='컨트리하우스')
        hotel2_inn = Hotel.objects.filter(place=result[1], classfication='여관')
        hotel2_villa = Hotel.objects.filter(place=result[1],
                                            classfication='빌라')
        hotel2_camping = Hotel.objects.filter(place=result[1],
                                              classfication='캠핑장')

        hotel3 = Hotel.objects.filter(place=result[2])
        hotel3_cost_up = hotel3.order_by('cost')
        hotel3_cost_down = hotel3.order_by('-cost')
        hotel3_rating_up = hotel3.order_by('rating')
        hotel3_rating_down = hotel3.order_by('-rating')
        hotel3_distance_up = hotel3.order_by('distance')
        hotel3_kind_up = hotel3.order_by('kind')
        hotel3_clean_up = hotel3.order_by('clean')
        hotel3_conv_up = hotel3.order_by('conv')
        hotel3_hotel = Hotel.objects.filter(place=result[2],
                                            classfication='호텔')
        hotel3_hostel = Hotel.objects.filter(place=result[2],
                                             classfication='호스텔')
        hotel3_guest = Hotel.objects.filter(place=result[2],
                                            classfication='게스트하우스')
        hotel3_apartment = Hotel.objects.filter(place=result[2],
                                                classfication='아파트')
        hotel3_apartmenthotel = Hotel.objects.filter(place=result[2],
                                                     classfication='아파트호텔')
        hotel3_motel = Hotel.objects.filter(place=result[2],
                                            classfication='모텔')
        hotel3_pension = Hotel.objects.filter(place=result[2],
                                              classfication='펜션')
        hotel3_resort = Hotel.objects.filter(place=result[2],
                                             classfication='리조트')
        hotel3_badandbreakfast = Hotel.objects.filter(place=result[2],
                                                      classfication='베드앤브렉퍼스트')
        hotel3_homestay = Hotel.objects.filter(place=result[2],
                                               classfication='홈스테이')
        hotel3_lodge = Hotel.objects.filter(place=result[2],
                                            classfication='롯지')
        hotel3_countryhouse = Hotel.objects.filter(place=result[2],
                                                   classfication='컨트리하우스')
        hotel3_inn = Hotel.objects.filter(place=result[2], classfication='여관')
        hotel3_villa = Hotel.objects.filter(place=result[2],
                                            classfication='빌라')
        hotel3_camping = Hotel.objects.filter(place=result[2],
                                              classfication='캠핑장')

        # restaurant1 = Restaurant.objects.filter(place=result[0])
        # restaurant2 = Restaurant.objects.filter(place=result[1])
        # restaurant3 = Restaurant.objects.filter(place=result[2])
        # restaurant4 = Restaurant.objects.filter(place=result[3])
        # restaurant5 = Restaurant.objects.filter(place=result[4])

        return render(
            request,
            'beer/ver3_result.html',
            {
                'login_session': login_session,
                'result': result,
                'hotels1': hotel1,
                'hotels2': hotel2,
                'hotels3': hotel3,
                'hotels4': hotel4,
                'hotels5': hotel5,
                # 'restaurant1': restaurant1,
                # 'restaurant2': restaurant2,
                # 'restaurant3': restaurant3,
                # 'restaurant4': restaurant4,
                # 'restaurant5': restaurant5,
            })
    else:
        return render(request, 'beer/ver3.html', context)
