import torch
import os
import numpy as np
import pickle
import torchvision.transforms as T
from torchvision import models

from tqdm import tqdm
from PIL import Image
from img_similarity_utils import plot_similar_images, write_similarImage_csv, compute_similar_images_pretrained, read_yaml_config_file, fit_2D_UMAP, fit_3D_UMAP


class FeatureCreator:
    '''
    This class contains the primary methods for using the pretrained models
    to extract the features for both input images and create the embedding for 
    the entire image set
    '''
    def __init__(self, cfg):
        
        self.config = read_yaml_config_file(cfg)
        self.init_general()
        
    def init_general(self):
        
        # Set parameters based on the desired pretrained model
        if self.config['ModelInfo']['modelName']=='resnet50':
            self.model = models.resnet50(pretrained=True)
            self.num_features = 2048
        elif self.config['ModelInfo']['modelName']=='resnet18':
            self.model = models.resnet18(pretrained=True)
            self.num_features = 512
        elif self.config['ModelInfo']['modelName']=='resnet101':
            self.model = models.resnet101(pretrained=True)
            self.num_features = 2048
        else:
            ValueError('No other models currently supported')
            
            
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), T.Resize(224)])
        self.feature_layer = self.model._modules.get('avgpool')    
        
        # Use GPU for processing if available
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'   
            
        if self.config['ModelInfo']['loadEmbedding']:
            embedding_img_names_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                                    self.config['ModelInfo']['embeddingImgNamesPath'])
                                                    
            with open(embedding_img_names_file, 'rb') as f:
                self.embedding_img_list = pickle.load(f)
                
            embedding_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                                    self.config['ModelInfo']['embeddingPath'])
            self.embedding = np.load(embedding_path)
            
            #Execute everything on cpu in test mode to save time
            self.device='cpu'
            
        self.model.to(self.device)
        self.model.eval()

        return None
    
    def compute_feature_vector(self, img):
        '''
        Compute the feature vector of the input image
        '''
      
        image = self.transform(img).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.num_features, 1, 1)

        def copy_data(m, i, o): embedding.copy_(o.data)

        h = self.feature_layer.register_forward_hook(copy_data)
        self.model(image)
        h.remove()
        
        return embedding.numpy()[0, :, 0, 0]

    def execute(self):
        '''
        Generate features for all the images and save results
        '''
        
        # Initialize
        all_vec_array = []
        all_vec_array_names = []
        all_vec_array_labels = []
        img_source_dir = self.config['DataLocationInfo']['imagesSourceDirectory']
        
        folders = self.__list_folders(img_source_dir)
        
        # Compute resnet features for all images in all folders
        for folder in folders:
            label = folder
            img_dir = os.path.join(img_source_dir, folder)
            img_files = self.__remove_png_strings(os.listdir(img_dir))
            
            for image in tqdm(img_files):
            
                I = Image.open(os.path.join(img_dir, image))
                vec = self.compute_feature_vector(I)
                all_vec_array.append(vec)
                all_vec_array_names.append(image)
                all_vec_array_labels.append(label)
                I.close() 
        
        numpy_embedding = np.asarray(all_vec_array)
        
        # Create output folders
        self.embedding_save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Embeddings')
        if not os.path.isdir(self.embedding_save_folder):
            os.mkdir(self.embedding_save_folder)
            
        np.save(os.path.join(self.embedding_save_folder,'data_embedding.npy'), numpy_embedding)
        with open(os.path.join(self.embedding_save_folder, 'embedding_image_file_names.pkl'), 'wb') as f:
            pickle.dump(all_vec_array_names, f)
           
        with open(os.path.join(self.embedding_save_folder, 'embedding_image_file_labels.pkl'), 'wb') as f:
            pickle.dump(all_vec_array_labels, f)
           
        fit_2D_UMAP(numpy_embedding, all_vec_array_labels, self.embedding_save_folder)
        fit_3D_UMAP(numpy_embedding, all_vec_array_labels, self.embedding_save_folder)
        
        return all_vec_array, all_vec_array_names, all_vec_array_labels, numpy_embedding
    
    def execute_evaluation(self):
        '''
        Evaluate for a single image. Return distances, indices of top K matches
        and names of top K matches
        '''
        I = Image.open(self.config['TestInfo']['testImagePath'])
        vec = self.compute_feature_vector(I)
        all_vec_array=[]
        all_vec_array.append(vec)
        vec = np.asarray(all_vec_array)
        
        #Check value of K before proceeding
        if self.config['TestInfo']['numK'] < 1:
            ValueError('K can not be < 1')
        elif self.config['TestInfo']['numK'] > 29997:
            ValueError('K can not be greater than all available images')
        
        distances, indices_list, similar_image_names = compute_similar_images_pretrained(vec, 
                                                            self.config['TestInfo']['numK'], 
                                                            self.embedding, 
                                                            self.embedding_img_list, 
                                                            self.device)
        
        plot_similar_images(self.config['TestInfo']['imageDataParentPath'], similar_image_names, self.config['TestInfo']['numK'], self.config['TestInfo']['testImagePath'])
        if self.config['TestInfo']['writeOutCsv']:
            write_similarImage_csv(similar_image_names, distances, self.config['TestInfo']['testImagePath'])
            
        return distances, indices_list, similar_image_names
    
    # Private Methods
    def __list_folders(self, directory):
        folders = []
        for entry in os.scandir(directory):
            if entry.is_dir():
                folders.append(entry.name)
                
        return folders
    
    def __remove_png_strings(self, strings):
        new_strings = []
        for string in strings:
            if not string.endswith(".png"):
                new_strings.append(string)
        return list(set(new_strings))