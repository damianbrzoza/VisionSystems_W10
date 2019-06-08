import cv2
from glob import glob
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import math
import numpy as np


def showing_two_images(im, im2):
    im = cv2.resize(im, (1024, 720))
    im2 = cv2.resize(im2, (1024, 720))
    cv2.imshow('wind1', im)
    cv2.imshow('wind2', im2)
    cv2.waitKey()


def get_files():
    list_of_files = glob('Zadanie 1/*.JPG')
    list_of_imgs = list()
    for file in list_of_files:
        list_of_imgs.append(cv2.imread(file))
    return list_of_imgs


def image_preprocessing(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    closed = 255 - closed
    return closed


def get_contours(im):
    new_contour = list()
    _, cnts, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour, h in zip(cnts, hierarchy[0]):
        if h[3] == -1 and h[2] != -1:
            new_contour.append(contour)
    return new_contour


def moments_df(list_contours):
    list_of_areas = list()
    list_of_arclength = list()
    list_of_circularity = list()
    list_of_df = list()
    for contour in list_contours:
        m = cv2.moments(contour)
        data = pd.DataFrame.from_dict(m, orient='index')
        list_of_df.append(data)
        list_of_areas.append(cv2.contourArea(contour))
        list_of_arclength.append(cv2.arcLength(contour, True))
        list_of_circularity.append(3.14 * cv2.contourArea(contour) /
                                   (cv2.arcLength(contour, True) * cv2.arcLength(contour, True)))

    data = pd.concat(list_of_df, axis=1)
    data = data.T
    data['area'] = list_of_areas
    data['arcLength'] = list_of_arclength
    data['circularity'] = list_of_circularity
    return data


def get_shorter(area, arclen):
    delta = pow(arclen/2, 2) - 4 * area
    b = ((arclen/2) + math.sqrt(delta))/2
    a = area/b
    return max([a, b])


def draw_by_class(contour_list, predicted_class, im):
    for contour, pred in zip(contour_list, predicted_class):
        if pred == 1:
            cv2.drawContours(im, contour, -1, (0, 255, 0), 3)
        elif pred == 2:
            cv2.drawContours(im, contour, -1, (255, 0, 0), 3)
        else:
            cv2.drawContours(im, contour, -1, (0, 0, 255), 3)


def draw_center_by_class(data, cnts, predicted_class, im):
    for [_, r], cnt, pred in zip(data.iterrows(), cnts, predicted_class):
        cx = int(r["m10"] / r["m00"])
        cy = int(r["m01"] / r["m00"])
        pixel = get_pixel()
        if pred == 1:
            cv2.circle(im, (cx, cy), 7, (0, 255, 0), -1)
            cv2.putText(im, str('{:.2f}'.format(2*math.sqrt(r['area']/3.14)/pixel)), (cx, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        elif pred == 2:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            area = cv2.contourArea(box)
            arclen = cv2.arcLength(box, True)
            lenght = get_shorter(area, arclen)/pixel

            cv2.drawContours(im, [box], 0, (255, 0, 0), 2)
            cv2.putText(im, str('{:.2f}'.format(lenght)), (cx, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

            cv2.circle(im, (cx, cy), 7, (255, 0, 0), -1)
        else:
            cv2.circle(im, (cx, cy), 7, (0, 0, 255), -1)
    im = cv2.resize(im, (1024, 720))
    cv2.imshow('wind1', im)
    cv2.waitKey()


def get_pixel():
    im = cv2.imread('Zadanie 2/kalib.JPG')
    im = cv2.Sobel(im, -1, 1, 0)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)
    list_of_distances = list()
    count = 0
    for number in im[0]:
        if number == 0:
            count += 1
        else:
            list_of_distances.append(count)
            count = 0
    list_of_distances = list(filter(lambda a: a > 100, list_of_distances))
    pixels = sum(list_of_distances) / len(list_of_distances)
    pixels_on_1mm = pixels / 4
    return pixels_on_1mm


def get_feature_info():
    ims = get_files()
    list_of_dfs = list()
    cnt = 0
    for im in ims:
        im = image_preprocessing(im)
        cnt = get_contours(im)
        data = moments_df(contours)
        list_of_dfs.append(data)

    cv2.drawContours(ims[-1], cnt, -1, (0, 255, 0), 3)
    cv2.imwrite('2.png', ims[-1])

    data = pd.concat(list_of_dfs)
    data = data.sort_values(by=['circularity'])
    data['class'] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    pd.set_option('display.max_columns', 500)

    x = data.iloc[:, :-1]  # independent columns
    y = data.iloc[:, -1]    # target column

    model = ExtraTreesClassifier()
    model.fit(x, y)
    print(model.feature_importances_)
    # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=x.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()
    print(data)


imgs = get_files()
i = 4
for image in imgs[:-1]:
    img = image_preprocessing(image)
    contours = get_contours(img)
    df = moments_df(contours)
    df1 = df[['circularity']]
    predict = list()
    for _, row in df1.iterrows():
        if row['circularity'] > 0.21000:
            predict.append(1)
        elif 0.18000 < row['circularity'] < 0.21000:
            predict.append(2)
        else:
            predict.append(0)
    draw_by_class(contours, predict, image)
    draw_center_by_class(df, contours, predict, image)
    cv2.imwrite(str(i)+'.png', image)
    i = i+1
