%% Constants for specific dataset used. 
NUM_FEMALE = 58;
NUM_MALE = 43; 

% The number of test per gender 
NUM_TEST = 15; 

% Training number per gender
NUM_F_TRAIN = NUM_FEMALE - NUM_TEST; 
NUM_M_TRAIN = NUM_MALE - NUM_TEST; 

CONTROL_INTENSITY = 5; 
TEST_INTENSITY = 0; 

% Create a Test and a Training Set 

training_set = []; 
training_labels = []; 

test_control_set = []; 
test_shadowed_set = []; 
testing_labels = []; 

female_indices = 1:NUM_FEMALE; 
male_indices = 1:NUM_MALE; 

% Indices used for training and testing 
f_train_indices = female_indices(1:NUM_FEMALE - NUM_TEST); 
f_test_indices = female_indices((NUM_FEMALE - NUM_TEST + 1):NUM_FEMALE);
m_train_indices = male_indices(1:NUM_MALE - NUM_TEST); 
m_test_indices = male_indices((NUM_MALE - NUM_TEST + 1):NUM_MALE);

% emotions = {'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'};  \
emotions = {'Happy', 'Sad'}; 
emotionLabels = 1:numel(emotions); 
baseFileName = 'generated_dataset/all_subject_images/'; 
% Loop through all emotions
for emotionsIndex = emotionLabels
    training_labels = [training_labels ones(1, NUM_F_TRAIN + NUM_M_TRAIN) * emotionsIndex]; 
    testing_labels = [testing_labels ones(1, NUM_TEST * 2) * emotionsIndex]; 
    emotion_str = emotions{emotionsIndex}; 
    
    for f_index = f_train_indices
        index_str = num2str(f_index); 
        if(f_index < 10)
            index_str = ['0' index_str]; 
        end
        file_name = [baseFileName 'F0' index_str '/' emotion_str '_SUN_' ...
            num2str(CONTROL_INTENSITY) '.png']; 
        im = preprocess(rgb2gray(imread(file_name)));
        [m,n] = size(im); 
        vector_im = reshape(im, [m*n, 1]); 
        training_set = [training_set, vector_im];  
    end
    
    for m_index = m_train_indices
        index_str = num2str(m_index); 
        if(m_index < 10)
            index_str = ['0' index_str]; 
        end
        file_name = [baseFileName 'M0' index_str '/' emotion_str '_SUN_' ...
            num2str(CONTROL_INTENSITY) '.png']; 
        im = preprocess(rgb2gray(imread(file_name)));
        [m,n] = size(im); 
        vector_im = reshape(im, [m*n, 1]); 
        training_set = [training_set, vector_im];  
    end
    
     for f_index = f_test_indices
        index_str = num2str(f_index); 
        if(f_index < 10)
            index_str = ['0' index_str]; 
        end
        file_name = [baseFileName 'F0' index_str '/' emotion_str '_SUN_' ...
            num2str(CONTROL_INTENSITY) '.png']; 
        shadowed_file_name = [baseFileName 'F0' index_str '/' emotion_str '_SUN_' ...
            num2str(TEST_INTENSITY) '.png']; 
        im = preprocess(rgb2gray(imread(file_name)));
        im_shadowed = preprocess(rgb2gray(imread(shadowed_file_name)));
        [m,n] = size(im); 
        vector_im = reshape(im, [m*n, 1]); 
        vector_im_shadowed = reshape(im_shadowed, [m*n, 1]);
        test_control_set = [test_control_set, vector_im]; 
        test_shadowed_set = [test_shadowed_set, vector_im_shadowed]; 
    end
    
    for m_index = m_test_indices
        index_str = num2str(m_index); 
        if(m_index < 10)
            index_str = ['0' index_str]; 
        end
        file_name = [baseFileName 'M0' index_str '/' emotion_str '_SUN_' ...
            num2str(CONTROL_INTENSITY) '.png']; 
        shadowed_file_name = [baseFileName 'M0' index_str '/' emotion_str '_SUN_' ...
            num2str(TEST_INTENSITY) '.png']; 
        im = preprocess(rgb2gray(imread(file_name)));
        im_shadowed = preprocess(rgb2gray(imread(shadowed_file_name)));
        [m,n] = size(im); 
        vector_im = reshape(im, [m*n, 1]); 
        vector_im_shadowed = reshape(im_shadowed, [m*n, 1]);
        test_control_set = [test_control_set, vector_im]; 
        test_shadowed_set = [test_shadowed_set, vector_im_shadowed];  
    end    
    emotionsIndex
end 

% Map every training into the fisher space 
[m_database, v_pca, v_fisher, projected_fisher_images, projected_pca_images] = ...
    FLD.FisherfaceCore(training_set, numel(emotions) , NUM_F_TRAIN + NUM_M_TRAIN); 

% Calculate Threshold 
min_proj = min(projected_fisher_images); 
max_proj = max(projected_fisher_images); 
true_happy = training_labels == 1;
true_sad = training_labels == 2; 
best_accuracy = 0; 
best_threshold = 0; 
all_good_thresholds = []; 

num_training = numel(training_labels); 
happy_mean = mean(projected_fisher_images(1:num_training/2)); 
sad_mean = mean(projected_fisher_images(num_training/2 + 1: num_training));

flip = false; 
if happy_mean > sad_mean
    projected_fisher_images = projected_fisher_images * -1; 
    min_proj = max_proj * -1;  
    max_proj = min_proj * -1;   
    flip = true; 
end

for candidate = min_proj:.02:max_proj
    pred_happy = projected_fisher_images < candidate; 
    pred_sad = projected_fisher_images >= candidate; 
    accuracy = double(sum(true_happy & pred_happy) + sum(true_sad & pred_sad)) ...
        / numel(training_labels); 
    if accuracy == 1.0
        all_good_thresholds = [all_good_thresholds candidate];
    elseif accuracy > best_accuracy
        best_threshold = candidate;
        best_accuracy = accuracy; 
    end
end

if numel(all_good_thresholds) > 0
    best_threshold = median(all_good_thresholds); 
end
    
% Test 
[~, num_tests] = size(test_control_set); 

%
% Uses the closest_neighbor_search
[control_predictions, projected_fisher_control, projected_pca_control] = FLD.Recognition(test_control_set, m_database, v_pca, v_fisher, ...
    projected_fisher_images, training_labels, best_threshold, flip); 
[test_predictions, projected_fisher_test, projected_pca_test] = FLD.Recognition(test_shadowed_set, m_database, v_pca, v_fisher, ...
    projected_fisher_images, training_labels, best_threshold, flip); 

% % Uses multi-functional svms 
% mdl = fitcecoc(projected_fisher_images', training_labels); 
% control_predictions = predict(mdl, projected_fisher_control')'; 
% test_predictions = predict(mdl, projected_fisher_test')'; 

histogram(projected_fisher_test(testing_labels==1), 20); 
figure
histogram(projected_fisher_test(testing_labels==2), 20); 
pause(); 

control_accuracy = double(sum(testing_labels == control_predictions))/num_tests 
test_accuracy = double(sum(testing_labels == test_predictions))/num_tests 

