from pretrained_model_feature_extractor import FeatureCreator

if __name__ == "__main__":
    
    config_file_name = 'ConfigurationFiles/testMainConfiguration.yaml'
    testObj = FeatureCreator(config_file_name)
    distances, indices_list, similar_image_names= testObj.execute_evaluation()

    #Print names out
    for idx in range(len(similar_image_names)-1):
        match_file_name = similar_image_names[idx].split('_')[0]
        print('Match '+str(idx+1)+': '+similar_image_names[idx])