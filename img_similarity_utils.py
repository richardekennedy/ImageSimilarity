import os
import csv
import shutil
import sklearn
import matplotlib.pyplot as plt
import PIL.Image
import yaml
from umap import UMAP


def read_yaml_config_file(config_file_path):
    '''
    This method reads in a YAML file
    '''
    
    fid = open(config_file_path, 'r')
    config = yaml.safe_load(fid)
    fid.close()
    return config


def compute_similar_images_pretrained(image_features, num_images, embedding, embedding_img_names, device):
    '''
    This method computes the similar images based on the cosine distance between the input image of interest
    and the features extracted and saved in the embedding generated using the ResNet-50 model

    '''
    image_embedding = image_features
    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=num_images+1, metric="cosine")
    embedding_to_fit_reshaped = embedding.reshape(embedding.shape[0], -1)
    
    #compute embedding
    knn.fit(embedding_to_fit_reshaped)
    distances, indices = knn.kneighbors(image_embedding)
    indices_list = indices.tolist()
    
    img_names = []
    for index in indices_list[0]:
        img_names.append(embedding_img_names[index])
    
    return distances, indices_list, img_names


def plot_similar_images(img_dir, similar_image_names, num_images, test_image_path):
    
    '''
    This method plots the similar images. Max 5 matches shown in the plot
    '''
    
    if num_images >=5:
        num_images = 5
    elif num_images < 1:
        ValueError('Choose K > 1')
                
    fig = plt.figure(figsize=(10, 5))
    for j in range(0,num_images+1):
        ax=[]
        if j == 0:
            new_name_parts = similar_image_names[j].split('_')
            img_name_parts = new_name_parts[1].split('.')
            new_path = os.path.join(img_dir, img_name_parts[0])
            new_image_name = new_name_parts[0]+'.jpg'
            
            if test_image_path is not None:
                img = PIL.Image.open(test_image_path)
            else:
                img = PIL.Image.open(os.path.join(new_path, new_image_name))
                
            ax = fig.add_subplot(1, num_images+1, 1)
            plt.title('Input Image \n '+ os.path.basename(test_image_path), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            new_name_parts = similar_image_names[j-1].split('_')
            img_name_parts = new_name_parts[1].split('.')
            new_path = os.path.join(img_dir, new_name_parts[0])
            new_image_name = new_name_parts[0] + '_' + img_name_parts[0]+'.jpg'
            
            img = PIL.Image.open(os.path.join(new_path, new_image_name))
            addAx = fig.add_subplot(1, num_images+1, j+1)
            addAx.set_xticks([])
            addAx.set_yticks([])
            ax.append(addAx)
            plt.title('Match '+str(j)+": \n"+new_image_name, fontsize=10)
            #ax.set_xticks([])
            #ax.set_yticks([])
        img = img.convert('RGB')
        plt.imshow(img)
        img.close()
    for axes in ax:
        axes.set_xticks([])
        axes.set_yticks([])
        
    outDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results')
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    
    plt.savefig(os.path.join(outDir, os.path.basename(test_image_path)+'_similar_images_fig.png'))    
    
    return None
        
def write_similarImage_csv(similar_image_names, distances, image_path):
    
    '''
    This method writes out the K top match file names, geological type, and rank out to a csv file
    
    '''
    
    csv_file_name = os.path.basename(image_path).split('.')[0] + '_similar_images.csv'
    csv_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')
    if not os.path.isdir(csv_folder):
        os.mkdir(csv_folder)
        
    csv_full_name = os.path.join(csv_folder, csv_file_name)
    
    with open(csv_full_name, 'w', newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(['Match Rank', 'Image Name'])
        
        for idx in range(len(similar_image_names)):
            row=[[idx+1, str(similar_image_names[idx])]]
            writer.writerows(row)
            
    return None


def fit_2D_UMAP(embedding, embedding_labels, out_plot_dir=None):
    '''
    This method fits a manifold and projects it into 2 dimensions using UMAP
    '''
    
    num_images = embedding.shape[0]
    flattened_embedding = embedding.reshape((num_images, -1))
    
    #% Here perform UMAP clustering
    umap_2d = UMAP(random_state=0)
    umap_2d.fit(flattened_embedding)
    
    print('Fitting 2D UMAP Projection\n')
    projections = umap_2d.transform(flattened_embedding)
    
    if out_plot_dir is not None:
        print('Plotting 2D UMAP Projection\n')
        
        cdict = {'banded': 'red',
                 'bubbly': 'blue', 
                 'grid': 'green', 
                 'lacelike':'black', 
                 'striped':'cyan'}
        A = []
        for label in embedding_labels:
            A.append(cdict[label])

        plt.scatter(projections[:,0], projections[:,1], c=A)
        plt.title("UMAP 2D Projection")
        plt.savefig(os.path.join(out_plot_dir, '2D_UMAP_Projection.png'))
        plt.close()

    return projections


def fit_3D_UMAP(embedding, embedding_labels, out_plot_dir=None):
    '''
    This method fits a manifold and projects it into 3 dimensions using UMAP
    '''
    num_images = embedding.shape[0]
    flattened_embedding = embedding.reshape((num_images, -1))
    
    #% Here perform UMAP clustering
    umap_3d = UMAP(n_components=3, random_state=0)
    proj_3D = umap_3d.fit_transform(flattened_embedding)
    print('Fitting 3D UMAP Projection\n')
    projections = umap_3d.transform(flattened_embedding)
    
    if out_plot_dir is not None:
        print('Plotting 3D UMAP Projection\n')
        
        cdict = {'banded': 'red',
                 'bubbly': 'blue', 
                 'grid': 'green', 
                 'lacelike':'black', 
                 'striped':'cyan'}
        A = []
        for label in embedding_labels:
            A.append(cdict[label])
            
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(proj_3D[:,0], proj_3D[:,1], proj_3D[:,2], c=A)
        plt.title("UMAP 3D Projection")
        plt.show()
        plt.savefig(os.path.join(out_plot_dir, '3D_UMAP_Projection.png'))
        plt.close()

    return projections


def copy_every_nth_file(source_dir, target_dir, n):
    
    '''
    Helper function to copy only part of files (copies every nth file)
    '''
    # Copy files to use from a larger set of data
    folders = []
    for entry in os.scandir(source_dir):
        if entry.is_dir():
            folders.append(entry.name)
            
    print(folders)
    
    for folder in folders:
        source_folder = os.path.join(source_dir, folder)
        target_folder = os.path.join(target_dir, folder)
        
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)
        
        files = os.listdir(source_folder)
        print('There are ' + str(len(files)) + ' files in the ' + folder + ' folder')
        
        for i, file_name in enumerate(files):
            if i % n == 0:
                source_path = os.path.join(source_folder, file_name)
                target_path = os.path.join(target_folder, file_name)
                shutil.copy2(source_path, target_path)
                print('Copying file ' + str(i) + ' of ' + str(len(files)))
