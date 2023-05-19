from pretrained_model_feature_extractor import FeatureCreator

if __name__ == "__main__":
    config_file_name = 'ConfigurationFiles/createEmbedding.yaml'
    testObj = FeatureCreator(config_file_name)
    all_vec_array, all_vec_array_names, all_vec_array_labels, numpy_embedding = testObj.execute()