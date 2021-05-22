import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from PIL import Image

# Generates a dateFrame from the given image file
# 3 columns to store RGB values & Each row contains data for a pixel 
def generate_df(i):
    img1 = plt.imread(i+'.jpg')
    dict = {'red':[], 'green':[], 'blue':[]}
    for row in img1:
        for pixel in row:
            dict['red'].append(pixel[0])
            dict['green'].append(pixel[1])
            dict['blue'].append(pixel[2])
    df = pd.DataFrame(dict,index=None)
    return df, img1

# Picks centroids for KMeans
def set_centroids(df,k):
    np.random.seed(k)
    centroid_index = np.random.randint(0,len(df),k)
    centroids = []
    for i in centroid_index:
        centroids.append(list(df.iloc[i]))
    print('centroids : ',centroids)
    return centroids

#To find Euclidean distance between two given lists
def distance(a,b):
    dist = 0
    for i,j in zip(a,b):
        dist += (i-j)**2
    return dist**(0.5)

def kMeans(df, iterations,centroids):
    iters = 0
    clusters = {}
    cluster_indexes = {}
    while(iters<iterations):
        for c in range(len(centroids)):
            clusters[c] = []
            cluster_indexes[c] = []
        for i in range(len(df)):
            if i%100000 == 0:
                print(i)
            dist_list = []
            point = list(df.iloc[i])
            for j in range(len(centroids)):
                dist_list.append(distance(point, centroids[j]))
            mini = min(dist_list)
            mini_index = dist_list.index(mini)
            clusters[mini_index].append(point)
            cluster_indexes[mini_index].append(i)
        new_centroids = []
        for c in range(len(centroids)):
            c_points = np.array(clusters[c])
            if len(clusters[c])>0:
                c_mean = np.sum(c_points, axis=0)/len(clusters[c])
                new_centroids.append(list(c_mean))
            else:
                new_centroids.append(centroids[c])
        centroids = new_centroids
        iters = iters + 1
    print('New centroids : ',centroids)
    for c in range(len(centroids)):
        for i in cluster_indexes[c]:
            df.at[i,'red'] = centroids[c][0]
            df.at[i,'green'] = centroids[c][1]
            df.at[i,'blue'] = centroids[c][2]
    return df
    
# Converts dataFrame to image
def save_image(df, img1, i, k):
    output = df.to_numpy()
    output = np.reshape(output, (img1.shape[0], img1.shape[1], img1.shape[2]))
    output = output.astype(np.uint8)
    plt.imshow(output) 
    plt.imsave(i+'_'+str(k)+'.png',output)
    plt.show() 
    
def main():
    iterations = 10
    for i in ['koala','penguins']:
        df, img1 = generate_df(i)
        df.to_csv(i+'.csv',index=None)
        for k in [20, 15, 10, 5, 2]:
            print('k : ',k)
            centroids = set_centroids(df,int(k))
            df = kMeans(df, iterations, centroids)
            save_image(df, img1,i,k)
main()